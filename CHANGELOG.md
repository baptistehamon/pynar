# Changelog

## v0.1.0 (unreleased)

### Bug fixes
- Fix `mask_uncomplete_years` to work with `np.datetime64` and `cftime.datetime` objects (issue [#18](https://forge.inrae.fr/agroclim/Indicators/OutilsPourIndicateurs/fonctionspython/pynar/-/issues/18), PR [#18](https://forge.inrae.fr/agroclim/Indicators/OutilsPourIndicateurs/fonctionspython/pynar/-/merge_requests/18)).

### Internal changes
- `xclim` dependency has been pinned to `<1.0.0` to avoid future issues with the release of `xclim v1.0.0` (issue [#27](https://forge.inrae.fr/agroclim/Indicators/OutilsPourIndicateurs/fonctionspython/pynar/-/work_items/27), PR [#19](https://forge.inrae.fr/agroclim/Indicators/OutilsPourIndicateurs/fonctionspython/pynar/-/merge_requests/19)).