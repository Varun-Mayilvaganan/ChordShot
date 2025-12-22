# ChordShot: Image-Based Context-Aware Music Generation

**Paper Overview and Technical Explanation**

This document provides a complete explanation of the ChordShot research project and paper. It is intended to help readers understand the **problem formulation, design choices, methodology, experiments, and findings** without needing to refer to the main repository README or inspect the code directly.

The accompanying paper is included in this repository and presents the same system in formal academic detail.

---

## 1. Problem Statement and Motivation

Music generation systems have seen significant progress with deep generative models, but most existing approaches rely on **text prompts, symbolic scores, or predefined musical conditions**. At the same time, images naturally convey rich contextual and emotional information through scene composition, color distribution, and objects present in the environment.

This project explores the following central question:

**Can semantic information extracted from static images be used to generate music that is both musically coherent and emotionally aligned with the visual content?**

ChordShot approaches this question by treating images as **creative conditioning signals** rather than descriptive prompts. Instead of asking users to describe music in words, the system attempts to infer musical intent directly from visual cues.

---

## 2. Core Idea of the Paper

The central idea of ChordShot is to introduce an **interpretable intermediate representation** between vision and music generation.

Rather than mapping images directly to audio, the system:

1. Extracts semantic visual features
2. Converts those features into musical attributes
3. Uses those attributes to condition a generative music model

This design choice prioritizes **interpretability, modularity, and controllability**, making it easier to analyze how visual features influence musical outcomes.

---

## 3. System Overview

The system is composed of four sequential stages:

1. **Scene Classification**
2. **Dominant Color Extraction**
3. **Object Detection**
4. **Conditional Music Generation**

Each stage contributes a different aspect of visual understanding that influences the final music output.

---

## 4. Scene Classification

Scene classification provides high-level contextual information about the environment depicted in the image (e.g., indoor, outdoor, natural, urban).

### Dataset

* Scene-15 dataset
* 15 scene categories including indoor rooms, natural environments, and urban settings

### Feature Representation

* Local texture features: DAISY descriptors
* Global structure features: Histogram of Oriented Gradients (HOG)
* DAISY descriptors clustered using MiniBatch K-Means
* Bag-of-Visual-Words (BoVW) representation
* Concatenation of BoVW and HOG features
* L2 normalization

### Classification Models

* Linear SVM
* RBF-kernel SVM

The RBF SVM achieved the highest performance (≈76.4% accuracy) and was selected for deployment.

### Role in Music Generation

Scene labels influence:

* Overall musical structure
* Genre tendencies
* Rhythmic density
* Ambient vs. rhythmic balance

---

## 5. Dominant Color Extraction

Color is treated as a primary carrier of emotional information.

### Method

* Image pixels clustered using K-Means (k = 4)
* RGB color space used for clustering
* Two most dominant clusters selected
* Colors matched to closest CSS3 color names using Euclidean distance

### Role in Music Generation

Dominant colors are mapped to:

* Emotional tone (calm, energetic, tense)
* Tempo range
* Harmonic brightness (major/minor tendencies)

For example:

* Warm colors → higher tempo, energetic textures
* Cool colors → slower tempo, ambient or atmospheric music

---

## 6. Object Detection

Object detection provides fine-grained semantic grounding.

### Model

* YOLOv8 (pretrained on COCO dataset)
* Medium and large variants used during experimentation
* Confidence threshold set to 0.25

### Output

* List of detected object labels per image

### Role in Music Generation

Objects influence:

* Instrument selection
* Textural elements
* Environmental sound metaphors

Examples:

* Natural elements → acoustic or ambient instruments
* Urban objects → percussive or electronic textures

---

## 7. Semantic Feature Encoding

The outputs from scene classification, color extraction, and object detection are combined into a structured semantic representation.

This representation encodes:

* Scene context
* Emotional tone
* Environmental semantics

These features are then mapped—using rule-based associations informed by prior affective studies—into musical attributes such as:

* Mood
* Tempo
* Instrumentation
* Texture

This step forms the **conceptual bridge** between vision and music.

---

## 8. Music Generation

### Model

* MusicGen-Small (transformer-based conditional music generator)

### Conditioning Strategy

* Visual semantics are converted into a natural-language music description
* The description acts as the conditioning input to MusicGen

Example structure:

* Mood descriptor
* Scene-inspired style
* Instrument list
* Environmental texture cues

### Output

* ~30-second audio clip
* 16 kHz sampling rate
* Saved as `.wav`

---

## 9. Experimental Observations

The evaluation focuses on qualitative and user-centered observations rather than large-scale quantitative metrics.

Key observations:

* Music reflects scene context more strongly when both color and object cues are included
* Color information plays a dominant role in emotional alignment
* Object grounding improves perceived realism and immersion
* Scene classification stabilizes musical structure

These findings support the hypothesis that visual semantics can meaningfully influence music generation.

---

## 10. Limitations

The paper explicitly acknowledges several limitations:

* Visual-to-music mappings are heuristic and not learned end-to-end
* MusicGen does not process images directly
* Fixed output length and sampling rate
* Generation latency depends on hardware resources

These constraints define the scope of the current work as **exploratory and system-oriented**.

---

## 11. Future Directions

Potential extensions discussed include:

* Learning visual–musical mappings via multimodal datasets
* End-to-end image-conditioned music models
* Longer and higher-fidelity compositions
* Interactive user feedback mechanisms
* Real-time audiovisual systems for games and installations
