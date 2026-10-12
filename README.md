# MP-CAS

Code for MP-CAS multimodal feature extraction and classification. It covers
screen-interaction, facial, audio-language and gaze features, as well as the
rapid-screening, diagnostic-aid and MCI-subtyping experiments.

## Installation

Use Python 3.11 for the complete workflow:

```bash
python -m pip install -e ".[face,gaze,audio]"
```

## Usage

Run `--help` on an entry point to see its required inputs:

```bash
mpcas extract-interaction --help
mpcas-facial infer --help
mpcas extract-audio --help
python -m mpcas_gaze.features --help
python -m mpcas_gaze.train --help
python -m mpcas_gaze.test --help
```

After extracting and combining features with `mpcas fuse-features` and
`mpcas prepare`, run the classification experiments:

```bash
mpcas run --data data/model_input.csv \
  --config configs/experiments.json --output outputs/main
```

The experiment profiles are `rapid_screening`, `diagnostic_aid` and
`mci_subtyping`. Their settings are in
[`configs/experiments.json`](configs/experiments.json).

## License

The source code in this repository is released under the MIT License. See the
[LICENSE](LICENSE) file for the full license text. Datasets, model weights and
third-party tools remain subject to their respective terms.
