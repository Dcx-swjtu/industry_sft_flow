# ScienceFlow CAD/Architecture SFT Bundle

This folder is a self-contained code bundle for building CAD and architecture
ScienceFlow SFT data from `image_caption`.

## Main Entry

```bash
cd /mnt/cpfs/chenxudu/workspace/workspace_sjtu/Idustry/datasets/industry_sft_flow
python run_scienceflow_sft.py run imgcap_02_architecture_b09db109d5 --config configs/image_caption_arch_cad_kimi_relay.yaml
```

## Included Code

- `run_scienceflow_sft.py`: CLI entrypoint.
- `configs/`: model and pipeline configs.
- `domain/`: typed pipeline result models.
- `operators/`: ScienceFlow stages.
- `prompts/`: prompts for each stage.
- `runner/`: retry, resume, and hard-gate export logic.
- `tools/`: sample building and export tools.
- `dataflow/`: bundled config/model/sample/prompt infrastructure.

## External Data Paths

- Input samples: `../data/image_caption_arch_cad`
- Source images: `../image_caption`
- Default run output: `runs/`
- Export target can be set with `tools/export_scienceflow_sft.py`.

Only `export_to_sft=true` runs should be exported for training.
