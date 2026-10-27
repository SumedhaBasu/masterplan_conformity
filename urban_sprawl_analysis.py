"""
Urban sprawl distance analysis for Circle-IV, Hyderabad.

This script compares distances from the starting-year urban footprint for:
  1. NU_to_U: non-urban at the start, urban at the end
  2. NU_to_NU: non-urban at the start, non-urban at the end

The script uses the raster grid directly. It does not create vector points.
"""

from pathlib import Path
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from scipy import ndimage
from scipy.stats import gaussian_kde, ks_2samp


# =============================================================================
# EDIT THIS CONFIGURATION SECTION BEFORE RUNNING
# =============================================================================
PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "Data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"

RASTER_FILES = {
    1995: DATA_FOLDER / "LULC_1995_C4_44N.tif",
    2005: DATA_FOLDER / "LULC_2005_C4_44N.tif",
    2015: DATA_FOLDER / "LULC_2015_C4_44N.tif",
    2025: DATA_FOLDER / "LULC_2025_C4_44N.tif",
}
BOUNDARY_FILE = DATA_FOLDER / "Circle-IV.shp"

BUILT_UP_CLASS = 1

SUBSAMPLE_SIZE = 50
KS_REPEATS = 200
RANDOM_SEED = 42

# True masks pixels outside Circle-IV as invalid. Set False only if the rasters
# have already been clipped exactly to Circle-IV and you do not want masking.
MASK_TO_CIRCLE = True
# =============================================================================


PERIODS = [(1995, 2005), (2005, 2015), (2015, 2025)]


def fail(message):
    """Raise a clear, user-facing error."""
    raise RuntimeError(message)

#Projection Checking

def check_projected_metric_crs(crs):
    """Require a projected CRS whose linear unit is metres."""
    if crs is None:
        fail("The rasters do not have a CRS. Assign a projected CRS in the source data first.")
    if not crs.is_projected:
        fail(
            f"The raster CRS is geographic ({crs}). Euclidean distance requires a "
            "projected CRS with metre units. Reproject the input rasters yourself; "
            "this script will not silently reproject or resample them."
        )
    units = str(getattr(crs, "linear_units", "")).lower()
    if units not in {"metre", "meter", "metres", "meters"}:
        fail(
            f"The raster CRS uses '{units or 'unknown'}' units, not metres. "
            "Use a projected CRS with metre units before running this script."
        )


def load_boundary(path, raster_crs):
    """Read Circle-IV and return geometries in the raster CRS."""
    if not path.exists():
        fail(f"Boundary shapefile was not found: {path}")
    boundary = gpd.read_file(path)
    if boundary.empty:
        fail(f"The boundary shapefile contains no features: {path}")
    if boundary.crs is None:
        fail("Circle-IV.shp has no CRS. Define its CRS before running the analysis.")
    if boundary.crs != raster_crs:
        print("Reprojecting the boundary only for masking; rasters are not reprojected.")
        boundary = boundary.to_crs(raster_crs)
    geometries = [geometry for geometry in boundary.geometry if geometry is not None]
    if not geometries:
        fail("Circle-IV.shp contains no usable geometries.")
    return geometries


def validate_rasters(raster_files):
    """Read rasters, validate their common grid, and return arrays and metadata."""
    arrays = {}
    metadata = None
    reference = None

    for year, path in raster_files.items():
        if not path.exists():
            fail(f"Input raster was not found for {year}: {path}")
        with rasterio.open(path) as src:
            if src.count < 1:
                fail(f"Raster has no bands: {path}")
            if src.crs is None:
                fail(f"Raster has no CRS: {path}")
            check_projected_metric_crs(src.crs)

            current = {
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "transform": src.transform,
                "x_size": abs(src.transform.a),
                "y_size": abs(src.transform.e),
                "nodata": src.nodata,
            }
            if current["x_size"] <= 0 or current["y_size"] <= 0:
                fail(f"Invalid pixel dimensions in {path}.")
            if not np.isclose(src.transform.b, 0) or not np.isclose(src.transform.d, 0):
                fail(
                    f"Raster {path} is rotated or sheared. This simple distance "
                    "workflow requires north-up rasters."
                )

            data = src.read(1)
            valid = np.isfinite(data)
            if src.nodata is not None:
                valid &= data != src.nodata
            arrays[year] = {"data": data, "valid": valid}

            if reference is None:
                reference = current
                metadata = src.meta.copy()
            else:
                if current["crs"] != reference["crs"]:
                    fail(f"CRS mismatch between rasters: {path}")
                if (current["width"], current["height"]) != (
                    reference["width"],
                    reference["height"],
                ):
                    fail(
                        f"Raster dimensions do not match for {path}. "
                        "Do not resample silently; align the source rasters first."
                    )
                if not np.allclose(
                    np.asarray(current["transform"]),
                    np.asarray(reference["transform"]),
                    rtol=0,
                    atol=1e-8,
                ):
                    fail(
                        f"Raster transform/grid does not match for {path}. "
                        "Do not resample silently; align the source rasters first."
                    )
                if not np.isclose(current["x_size"], reference["x_size"]) or not np.isclose(
                    current["y_size"], reference["y_size"]
                ):
                    fail(f"Pixel size does not match for {path}.")

    print(
        f"Validated common grid: {reference['width']} columns x "
        f"{reference['height']} rows, pixel size "
        f"{reference['x_size']:.3f} x {reference['y_size']:.3f} metres."
    )
    return arrays, metadata, reference


def write_raster(path, values, metadata, nodata, dtype):
    """Write one-band output while preserving the input grid."""
    profile = metadata.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(dtype), 1)


def valid_for_year(year_data, study_mask):
    return year_data["valid"] & study_mask


def ks_subsampled(a, b, n=50, repeats=200, seed=42):
    """
    Run repeated, without-replacement two-sample KS tests.

    If either group has fewer than n observations, the smaller group size is
    used for both groups. The returned p-values are exploratory sensitivity
    results and are not a formal inferential probability.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        fail("KS testing needs at least one valid distance in both groups.")
    actual_n = min(n, a.size, b.size)
    if actual_n < 2:
        fail("KS testing needs at least two observations in each group.")

    full = ks_2samp(a, b)
    rng = np.random.default_rng(seed)
    d_values = np.empty(repeats, dtype=float)
    p_values = np.empty(repeats, dtype=float)
    for i in range(repeats):
        a_sample = a[rng.choice(a.size, size=actual_n, replace=False)]
        b_sample = b[rng.choice(b.size, size=actual_n, replace=False)]
        result = ks_2samp(a_sample, b_sample)
        d_values[i] = result.statistic
        p_values[i] = result.pvalue

    return {
        "n_new": int(a.size),
        "n_nonurban": int(b.size),
        "subsample_n": int(actual_n),
        "D_full": float(full.statistic),
        "p_full": float(full.pvalue),
        "D_mean": float(d_values.mean()),
        "D_std": float(d_values.std(ddof=1)),
        "p_mean": float(p_values.mean()),
        "p_std": float(p_values.std(ddof=1)),
        "p_lt_0_05_percent": float((p_values < 0.05).mean() * 100),
        "D_values": d_values,
        "p_values": p_values,
    }


def make_ecdf_plot(period, new_distances, nonurban_distances, ks_result):
    """Save an ECDF comparison plot for one period."""
    start, end = period
    plt.figure(figsize=(7, 5))
    for values, label, colour in (
        (new_distances, "NU_to_U", "tab:red"),
        (nonurban_distances, "NU_to_NU", "tab:blue"),
    ):
        sorted_values = np.sort(values) / 1000.0
        y = np.arange(1, sorted_values.size + 1) / sorted_values.size
        plt.step(sorted_values, y, where="post", label=f"{label} (n={values.size})", color=colour)
    plt.xlabel("Distance from starting urban footprint (km)")
    plt.ylabel("Empirical cumulative probability")
    plt.title(f"Distance ECDF: {start}-{end}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.text(
        0.98,
        0.05,
        f"Full-sample KS D = {ks_result['D_full']:.3f}",
        transform=plt.gca().transAxes,
        ha="right",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"ecdf_{start}_{end}.png", dpi=180)
    plt.close()


def make_pdf_plot(period, new_distances, nonurban_distances):
    """Save normalized histograms and optional KDE curves for one period."""
    start, end = period
    groups = (
        (new_distances / 1000.0, "NU_to_U", "tab:red"),
        (nonurban_distances / 1000.0, "NU_to_NU", "tab:blue"),
    )
    plt.figure(figsize=(7, 5))
    for values, label, colour in groups:
        plt.hist(
            values,
            bins=30,
            density=True,
            alpha=0.25,
            color=colour,
            edgecolor=colour,
            label=f"{label} histogram",
        )
        if values.size >= 2 and np.unique(values).size >= 2:
            x_values = np.linspace(values.min(), values.max(), 300)
            density = gaussian_kde(values)(x_values)
            plt.plot(x_values, density, color=colour, linewidth=2, label=f"{label} KDE")
        else:
            print(f"Warning: KDE skipped for {label} in {start}-{end}; insufficient variation.")
    plt.xlabel("Distance from starting urban footprint (km)")
    plt.ylabel("Probability density")
    plt.title(f"Distance PDF: {start}-{end}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"pdf_{start}_{end}.png", dpi=180)
    plt.close()


def make_ks_histogram(all_ks_results):
    """Save side-by-side histograms of subsampled KS statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, (period, result) in zip(axes, all_ks_results.items()):
        start, end = period
        values = result["D_values"]
        axis.hist(values, bins=15, color="slateblue", alpha=0.8, edgecolor="white")
        axis.axvline(result["D_mean"], color="black", linestyle="--", label="Mean D")
        axis.axvline(
            result["D_mean"] - result["D_std"],
            color="black",
            linestyle=":",
            label="Mean +/- SD",
        )
        axis.axvline(result["D_mean"] + result["D_std"], color="black", linestyle=":")
        axis.set_title(f"{start}-{end}")
        axis.set_xlabel("KS D statistic")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Number of repetitions")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        f"Subsampled KS statistics (n={SUBSAMPLE_SIZE}, repetitions={KS_REPEATS})"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "ks_D_distributions.png", dpi=180)
    plt.close(fig)


def make_pvalue_plot(all_ks_results):
    """Save an optional exploratory p-value sensitivity plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, (period, result) in zip(axes, all_ks_results.items()):
        start, end = period
        axis.hist(result["p_values"], bins=15, color="darkorange", alpha=0.8, edgecolor="white")
        axis.axvline(0.05, color="black", linestyle="--", label="p = 0.05")
        axis.set_title(f"{start}-{end}")
        axis.set_xlabel("Subsampled p-value")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Number of repetitions")
    axes[0].legend(fontsize=8)
    fig.suptitle("Exploratory p-value sensitivity analysis (not definitive inference)")
    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER / "p_value_sensitivity.png", dpi=180)
    plt.close(fig)


def write_readme():
    text = f"""Urban sprawl distance analysis
================================

Configuration used:
- Built-up class code: {BUILT_UP_CLASS}
- Subsample size requested: {SUBSAMPLE_SIZE}
- KS repetitions: {KS_REPEATS}
- Random seed: {RANDOM_SEED}
- Circle-IV masking enabled: {MASK_TO_CIRCLE}

Workflow:
1. The four LULC rasters are checked for a common projected, metre-based CRS,
   dimensions, pixel size, transform, and grid alignment.
2. Urban masks, urban-edge rasters, transition masks, and distance rasters are
   written without resampling. Distances use the actual x and y pixel sizes.
3. Distance is measured to the nearest urban pixel centre using
   scipy.ndimage.distance_transform_edt. It is not measured to a polygon edge.
4. NU_to_U means non-urban at the start and urban at the end. NU_to_NU means
   non-urban at both dates. Only pixels valid in both rasters are used.
5. PDF plots contain normalized distance histograms and smooth Gaussian KDE
   curves where the group has enough distinct values.

Statistical limitations:
- The full-sample KS result is provided for reference.
- Repeated random subsampling is an exploratory sensitivity analysis. The
  repeated p-values, their mean, and the percentage below 0.05 are not a
  formal probability that the distributions differ.
- Random subsampling does not fix spatial autocorrelation. Nearby pixels may
  be dependent and random samples can still contain spatial clusters.
- No master-plan zones are analysed in this version.

To run:
  python urban_sprawl_analysis.py
"""
    (OUTPUT_FOLDER / "README.txt").write_text(text, encoding="utf-8")


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print("Reading and validating input rasters...")
    arrays, metadata, reference = validate_rasters(RASTER_FILES)
    geometries = load_boundary(BOUNDARY_FILE, reference["crs"])

    study_mask = np.ones((reference["height"], reference["width"]), dtype=bool)
    if MASK_TO_CIRCLE:
        study_mask = geometry_mask(
            geometries,
            out_shape=study_mask.shape,
            transform=reference["transform"],
            invert=True,
        )

    valid_masks = {
        year: valid_for_year(arrays[year], study_mask) for year in RASTER_FILES
    }
    urban_masks = {}
    urban_valid_masks = {}
    for year in RASTER_FILES:
        urban = arrays[year]["data"] == BUILT_UP_CLASS
        valid = valid_masks[year]
        urban_masks[year] = urban & valid
        urban_valid_masks[year] = valid
        if not np.any(urban_masks[year]):
            fail(
                f"No pixels use BUILT_UP_CLASS={BUILT_UP_CLASS} in {year}. "
                "Check the class code in the configuration section."
            )
        output = np.where(valid, urban_masks[year], 255)
        write_raster(
            OUTPUT_FOLDER / f"urban_mask_{year}.tif",
            output,
            metadata,
            nodata=255,
            dtype="uint8",
        )
        print(f"Saved urban mask for {year}.")

    print("Creating urban-edge rasters...")
    for year in (1995, 2005, 2015):
        inner_urban = ndimage.binary_erosion(
            urban_masks[year], structure=np.ones((3, 3), dtype=bool), border_value=0
        )
        edge = urban_masks[year] & ~inner_urban
        output = np.where(valid_masks[year], edge, 255)
        write_raster(
            OUTPUT_FOLDER / f"urban_edge_{year}.tif",
            output,
            metadata,
            nodata=255,
            dtype="uint8",
        )

    all_ks_results = {}
    csv_rows = []
    distributions = {}
    print("Creating distance rasters, transition masks, plots, and KS results...")
    for start, end in PERIODS:
        start_valid = valid_masks[start]
        end_valid = valid_masks[end]
        transition_valid = start_valid & end_valid
        start_urban = urban_masks[start]
        end_urban = urban_masks[end]

        nonurban_start = ~start_urban
        new_urban = transition_valid & nonurban_start & end_urban
        remaining_nonurban = transition_valid & nonurban_start & ~end_urban

        transition_outputs = {
            "NU_to_U": new_urban,
            "NU_to_NU": remaining_nonurban,
        }
        for label, mask in transition_outputs.items():
            output = np.where(transition_valid, mask, 255)
            write_raster(
                OUTPUT_FOLDER / f"transition_{label}_{start}_{end}.tif",
                output,
                metadata,
                nodata=255,
                dtype="uint8",
            )

        distance_pixels = ndimage.distance_transform_edt(
            ~start_urban,
            sampling=(reference["y_size"], reference["x_size"]),
        )
        distance_valid = start_valid
        distance_output = np.where(distance_valid, distance_pixels, -9999.0)
        write_raster(
            OUTPUT_FOLDER / f"distance_{start}_{end}_m.tif",
            distance_output,
            metadata,
            nodata=-9999.0,
            dtype="float32",
        )

        dist_new = distance_pixels[new_urban]
        dist_nonurban = distance_pixels[remaining_nonurban]
        distributions[f"dist_new_{str(start)[-2:]}_{str(end)[-2:]}"] = dist_new
        distributions[f"dist_nonurban_{str(start)[-2:]}_{str(end)[-2:]}"] = dist_nonurban

        result = ks_subsampled(
            dist_new,
            dist_nonurban,
            n=SUBSAMPLE_SIZE,
            repeats=KS_REPEATS,
            seed=RANDOM_SEED,
        )
        all_ks_results[(start, end)] = result
        csv_rows.append(
            {
                "period": f"{start}-{end}",
                **{key: value for key, value in result.items() if key not in {"D_values", "p_values"}},
            }
        )
        make_ecdf_plot((start, end), dist_new, dist_nonurban, result)
        make_pdf_plot((start, end), dist_new, dist_nonurban)
        print(
            f"{start}-{end}: NU_to_U={dist_new.size}, NU_to_NU={dist_nonurban.size}, "
            f"subsample n={result['subsample_n']}, "
            f"D_full={result['D_full']:.4f}, p_full={result['p_full']:.4g}"
        )

    np.savez_compressed(OUTPUT_FOLDER / "distance_distributions.npz", **distributions)
    pd.DataFrame(csv_rows).to_csv(OUTPUT_FOLDER / "ks_results.csv", index=False)
    make_ks_histogram(all_ks_results)
    make_pvalue_plot(all_ks_results)
    write_readme()
    print(f"Done. Outputs were written to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
