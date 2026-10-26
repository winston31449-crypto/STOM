# Source-Traceable Operator Mixtures for Long-Horizon Time Series Forecasting

## Introduction

Long-horizon forecasting methods have advanced in predictive accuracy, yet the numerical contributions forming each future prediction often remain implicit. STOM introduces a future-formation perspective that constructs forecasts from explicit, horizon- and variable-dependent temporal operator contributions. These forward-defined contributions make forecast formation directly traceable and provide a basis for diagnosis and targeted improvement.

## Overall Architecture

STOM first decomposes the historical input into smooth and residual components. Four temporal paths—phase, smooth, local, and spectral—then generate candidate forecasts. A horizon-aware gating mechanism organizes these paths into signed contributions, whose additive combination forms the final prediction. The spectral contribution can be further decomposed into model-defined contributions from different historical residual positions.

## Highlights

- **Future formation:** Horizon- and variable-dependent routing combines four temporal paths to form each forecast.
- **Traceable contributions:** Realized signed operator contributions reconstruct the forecast, and the spectral path traces individual historical residual sources.
- **Formation diagnosis:** Forward-defined contributions support retrospective error analysis, operator removal, shrinkage, and joint adjustment.
- **Forecasting performance:** STOM improves average MSE over the matched LinAttn+RoPE backbone on seven forecasting datasets.
- **Targeted improvement:** Validation-selected adjustments of exposed contributions improve held-out forecasting accuracy.

## Code

This repository contains the main components of STOM, together with selected experimental and analysis code used in this study.

## Experimental Setup

<table>
  <thead>
    <tr>
      <th><div align="center">Item</div></th>
      <th><div align="center">Setting</div></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Framework</td>
      <td align="center">PyTorch</td>
    </tr>
    <tr>
      <td align="center">Hardware</td>
      <td align="center">A single NVIDIA H20 GPU with 95 GB memory</td>
    </tr>
    <tr>
      <td align="center">Datasets</td>
      <td align="center">Weather, Solar-Energy, Electricity, ETTh1, ETTh2, ETTm1, and ETTm2</td>
    </tr>
    <tr>
      <td align="center">Input length</td>
      <td align="center">96</td>
    </tr>
    <tr>
      <td align="center">Prediction horizons</td>
      <td align="center">96, 192, 336, and 720</td>
    </tr>
    <tr>
      <td align="center">Evaluation metrics</td>
      <td align="center">MSE and MAE</td>
    </tr>
  </tbody>
</table>
