# Changelog

All notable changes to the In-Car Violence Detection System will be documented in this file.

## [1.0.0] - 2026-05-20

### Added
- Real-time weapon detection using YOLOv8n (knives, guns, sharp objects)
- Real-time violence detection using VD-MIL (MoViNet A0 + MIL classifier)
- Integrated threat evaluation system with 4 levels (Normal/Watch/Warning/Critical)
- Alert system with debouncing (3-second cooldown)
- Temporal smoothing for violence scores (30-frame buffer)
- Support for webcam and video file input
- Annotated video output with bounding boxes, meters, and banners
- SCVD dataset evaluation pipeline
- Console logging with color-coded threat levels

### Models
- **Weapon Model**: YOLOv8n, trained on custom dataset, ~6MB
- **Violence Model**: VD-MIL, MoViNet A0 backbone + MIL classifier, ~11MB per checkpoint
  - Best checkpoint: model_12.pt (loss: 0.0132)
  - 14 epochs trained (interrupted but converged)

### Performance
- SCVD Test Set (58 videos):
  - Accuracy: 82.8%
  - Precision: 62.5%
  - Recall: 41.7%
  - F1-Score: 50.0%
  - AUC-ROC: 0.7554
- Real-time inference: ~30fps at 1280×720 on GTX 1650

### Technical Stack
- PyTorch 2.2.1 + CUDA 11.8
- Ultralytics YOLOv8 8.1.0
- OpenCV 4.8.1.78
- NumPy 1.24.4 (locked - 2.x breaks PyTorch)
- Python 3.x

### Known Limitations
- Low recall (41.7%) - misses some violence videos
- Only 14 epochs trained (interrupted, could improve with more)
- Small test set (58 videos) - limited for robust evaluation
- Frame size locked to 172×172 for VD-MIL
- Violence model trained on CCTV footage, may not generalize perfectly to in-car scenarios
- No REST API or database persistence
- Single camera support only

### Documentation
- README.md with installation and usage
- Architecture diagram and data flow
- Model specifications and performance benchmarks
- Troubleshooting guide
- Training pipeline reference

---

## [Future] Version 2.0 Roadmap

### Planned Improvements
- [ ] Retrain violence model to 30+ epochs for higher accuracy
- [ ] Increase test set size for robust evaluation
- [ ] Add data augmentation for better generalization
- [ ] Hyperparameter tuning (learning rate, batch size)
- [ ] TensorRT optimization for >60fps inference
- [ ] Model quantization for edge deployment
- [ ] Multi-camera support
- [ ] REST API for integration
- [ ] Alert persistence (database logging)
- [ ] Docker containerization
- [ ] Edge deployment (Jetson Nano / Raspberry Pi)
- [ ] Web dashboard for monitoring

---

## Versioning
We use [Semantic Versioning](https://semver.org/):
- MAJOR: Incompatible API changes
- MINOR: Added functionality (backward compatible)
- PATCH: Bug fixes (backward compatible)
