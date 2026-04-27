# Example Evaluation Results

All results are reported with random seed 4242. However, as many baselines rely on VLMs (e.g., GPT and Gemini), some variability across runs may be observed.

### Object Navigation (AI2-THOR)

| Models                          | SR% ↑ | SPL% ↑ | SEL% ↑ | ToO ↓ | Token (/step) ↓ |
|---------------------------------|------:|-------:|-------:|------:|----------------:|
| **Reconstruction-based 3D Cache** |       |        |        |       |                 |
| VGGT (w/ frontier)              | 57.50 | 54.92  | 49.51  | 5.75  | **0**           |
| VGGT (w/ GPT-5m)                | 52.50 | 51.67  | 47.17  | 5.87  | 795.52          |
| VGGT (w/ Gemini 3.0)            | 59.46 | 57.21  | 52.69  | 5.48  | 2825.49         |
| **Generative Scene Reconstruction** |   |        |        |       |                 |
| DFoT-VGGT (w/ GPT-5m)           | 45.00 | 44.17  | 43.42  | 4.37  | 809.41          |
| DFoT-VGGT (w/ Gemini 3.0)       | 56.41 | 55.31  | 52.26  | 3.18  | 2909.21         |
| NWM-VGGT (w/ GPT-5m)            | 60.00 | 56.70  | 54.99  | 3.62  | 614.89          |
| NWM-VGGT (w/ Gemini 3.0)        | 60.00 | 55.38  | 47.57  | **3.17** | 311.93       |
| **VLM Agents**                  |       |        |        |       |                 |
| GPT-5m                          | 47.50 | 35.72  | 40.44  | 8.26  | 6531.00         |
| Gemini 3.0                      | 72.50 | 49.35  | 53.86  | 5.42  | 7937.84         |
| Qwen3-VL-8B-Instruct            | 27.50 | 19.27  | 24.46  | 10.42 | 221.36          |
| **3D-Belief**                   | **85.0** | **63.90** | **63.09** | 3.30 | **0** |

### 3D Contextual Reasoning (3D-CORE)

| Models        | Visibility | BEV IoU ↑ | 3D IoU ↑ | Chamfer ↓ | SigLIP ↑ | Recognition ↑ |
|---------------|-----------:|----------:|---------:|----------:|---------:|--------------:|
| DFoT-VGGT     | 0.05       | 0.110     | 0.064    | 2.681     | 0.265    | 0.126         |
| DFoT-VGGT     | 0.55       | 0.362     | 0.243    | 0.830     | 0.798    | 0.767         |
| DFoT-VGGT     | 0.95       | 0.372     | 0.242    | 0.189     | 0.857    | 0.838         |
| **3D Belief** | 0.05       | **0.147** | **0.083**| **2.435** | **0.329**| **0.165**     |
| **3D Belief** | 0.55       | **0.484** | **0.318**| **0.216** | **0.855**| **0.930**     |
| **3D Belief** | 0.95       | **0.535** | **0.369**| **0.187** | **0.884**| **0.909**     |