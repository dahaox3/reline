# Reline fork for ComicRead integration

This fork extends [rewaifu/reline](https://github.com/rewaifu/reline) with features used by the local GUI and ComicRead integration.

## New features

- `folder_reader` and `file_reader` support `dynamic` mode, preserving each image as gray or RGB instead of forcing one format for the whole batch.
- `upscale` can automatically detect color pages and select different models for color and gray images.
- `upscale` supports `color_detect_mode` values: `auto`, `force_color`, and `force_gray`.
- `upscale` supports `model_cache_mode`: `low_memory` loads one model at a time, while `high_memory` can keep switched models cached.
- `level` and `halftone` support `skip_on_color`, allowing color pages to bypass monochrome-oriented post-processing.
- `snapshot_writer` can save an intermediate image state without stopping the rest of the pipeline.
- `api_output` marks which image state should be returned by API clients while later nodes may still continue processing.

## Usage

Generate config via configurator: https://configurator.yor.ovh/

```shell
pip install reline
reline -c config.json
```

To install this fork directly:

```shell
pip install git+https://github.com/dahaox3/reline.git
```
