# Datasets

The event logs are public but are **not** redistributed here. Download each file, place it at the path below, and check it:

```powershell
python datasets/verify_checksums.py
```

| Log | Role in the paper | DOI | License | Place the file at |
|---|---|---|---|---|
| BPI 2020 International Declarations | Development log | [10.4121/uuid:2bbf8f6a-fc50-48eb-aa9e-c4ea5ef7e8c5](https://doi.org/10.4121/uuid:2bbf8f6a-fc50-48eb-aa9e-c4ea5ef7e8c5) | CC BY-NC 4.0 | `raw/bpi2020_international_declarations/InternationalDeclarations.xes.gz` |
| BPI 2017 | Evaluation | [10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b) | 4TU General Terms of Use | `raw/bpi2017/BPI Challenge 2017.xes.gz` |
| BPI 2019 | Evaluation | [10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1](https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1) | CC BY 4.0 | `raw/bpi2019/BPI_Challenge_2019.xes` |
| Sepsis Cases | Evaluation (untuned) | [10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460) | 4TU General Terms of Use | `raw/sepsis/Sepsis Cases - Event Log.xes.gz` |
| Road Traffic Fine Management | Evaluation (untuned) | [10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5) | 4TU General Terms of Use | `raw/rtfm/Road_Traffic_Fine_Management_Process.xes.gz` |
| BPI 2012 | Supplementary | [10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f) | 4TU General Terms of Use | `raw/bpi2012/BPI_Challenge_2012.xes.gz` |
| BPI 2015 Municipality 1 | Supplementary | [10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17](https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17) | 4TU General Terms of Use | `raw/bpi2015_municipality1/BPIC15_1.xes` |

Read the license on each landing page before use; BPI 2020 does not allow commercial use. Some downloads are zipped or named differently; extract or rename them so that the path and SHA-256 match `metadata/checksums.sha256`.

## Folders

| Folder | Contents |
|---|---|
| `metadata/datasets.yaml` | DOI, license, raw path, `extract-variants` options and expected case, event, activity and variant counts per log |
| `metadata/checksums.sha256` | SHA-256 of each raw file as used in the paper |
| `raw/` | Downloaded logs (ignored by git) |
| `processed/` | Output of `extract-variants` (ignored by git) |
| `interim/` | LLM response cache and other intermediate files (ignored by git) |
