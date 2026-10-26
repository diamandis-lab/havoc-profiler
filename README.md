# Havoc-Profiler

Histopathology Foundation Model Framework for Refined Risk Stratification in Diffuse Gliomas

1. **Region Delineation:** [HAVOC](https://www.science.org/doi/10.1126/sciadv.adg1894)
2. **Histopathological Classification:** `VGG19` + `XGBoost`
3. **Feature Extraction:** [Prov-GigaPath](https://huggingface.co/prov-gigapath/prov-gigapath)
4. **Risk Inference:** `XGBoost` (Survival Analysis)
5. **Histomorphometric Analysis:** [HoverNet-Kumar](https://github.com/vqdang/hover_net) + [CellProfiler](https://cellprofiler.org/)

This repository contains the official codebase and inference pipelines for the Havoc-Profiler methodology, accompanying the paper:

> Alvi, A., Leon, A. J., Oreopoulos, D. G., Babei, P., Paliwal, A., Hu, J., Kamski-Hennekam, E., & Diamandis, P. (n.d.).  
> *Refined risk stratification in diffuse gliomas using histopathology foundation models* [Manuscript submitted for publication].

---

## Live Application

A web-hosted version is deployed on CODIDO for analysis of WSIs of diffuse gliomas:

- **Launch:** [Havoc-Profiler on CODIDO](https://www.codido.co/marketplace/browse?RepoInfo%5Bquery%5D=HAVOC-Profiler)

---

## Docker Setup & Execution

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (v20.10+)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (required for GPU acceleration via `--gpus all`)

---

### End-to-End Workflow

1. Clone repository and build the Docker image
```bash
git clone https://github.com/diamandis-lab/havoc-profiler.git
cd havoc-profiler
docker build -t havoc-profiler:latest .
```

2. Initialize host mount points and stage input WSI (.svs or .ndpi)
```bash
mkdir -p /data/havoc_profiler/svs_files/codido
mkdir -p /data/havoc_profiler_study
cp /path/to/TCGA-06-0143-01Z-00-DX1.svs /data/havoc_profiler/svs_files/codido/
```
3. Launch workflow (preserves user permissions and mounts volumes)
```bash
docker run -it --rm \
  --gpus all \
  -u $(id -u):$(id -g) \
  -v /data/havoc_profiler:/data/havoc_profiler \
  -v /data/havoc_profiler_study:/data/havoc_profiler_study \
  --entrypoint bash \
  havoc-profiler:latest -c "python3 /app/src/app_codido.py \
    --input TCGA-06-0143-01Z-00-DX1.svs \
    --output test1 \
    --k_value 6 \
    --glioma_subtype IDHwt"
```

---

### Pipeline Arguments

| Argument | Type | Allowed Values | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `--input` | `str` | Any valid filename | Filename of the WSI inside `/data/havoc_profiler/svs_files/codido/` | `TCGA-06-0143-01Z-00-DX1.svs` |
| `--output` | `str` | Any valid string | Name of the output subdirectory for results and logs | `test1` |
| `--k_value` | `int` | Integer (e.g., `4`–`10`) | Number of clusters / latent space partitions to extract | `6` |
| `--glioma_subtype` | `str` | `IDHwt`<br>`IDHmut_noncodel`<br>`IDHmut_codel`<br>`unknown` | Molecular subtype label. If "Unknown" is selected, the risk inference step is skipped. | `IDHwt` |

## Citation

If you use this software or find our work helpful in your research, please cite:

```bibtex
@unpublished{alvi_refined_gliomas,
  author = {Alvi, Anzar and Leon, Alberto J. and Oreopoulos, Dimitrios G. and Babei, Parsa and Paliwal, Ameesha and Hu, Jinzhen and Kamski-Hennekam, Evelyn and Diamandis, Phedias},
  title  = {Refined risk stratification in diffuse gliomas using histopathology foundation models},
  note   = {Manuscript submitted for publication}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
