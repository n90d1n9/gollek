#!/bin/bash

# Test Data Download Script for Multimodal Testing
# This script downloads and prepares test datasets for integration testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/src/test/resources"
IMAGES_DIR="${DATA_DIR}/images"
DOCUMENTS_DIR="${DATA_DIR}/documents"
DATASETS_DIR="${DATA_DIR}/datasets"

echo "========================================="
echo "Multimodal Test Data Download Script"
echo "========================================="
echo ""

# Create directories
echo "Creating directories..."
mkdir -p "${IMAGES_DIR}"
mkdir -p "${DOCUMENTS_DIR}"
mkdir -p "${DATASETS_DIR}"

# Function to download file with progress
download_file() {
    local url="$1"
    local output="$2"
    local description="$3"
    
    echo "Downloading ${description}..."
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "${output}" "${url}"
    elif command -v curl &> /dev/null; then
        curl -L -# -o "${output}" "${url}"
    else
        echo "Error: Neither wget nor curl available"
        return 1
    fi
    echo "✓ Downloaded: ${output}"
}

# Download test images
echo ""
echo "Downloading test images..."
echo "-------------------------------------------"

# Test image 1: Object (car)
download_file \
    "https://picsum.photos/seed/car123/512/512.jpg" \
    "${IMAGES_DIR}/test-car.jpg" \
    "Car image"

# Test image 2: Animal (dog)
download_file \
    "https://picsum.photos/seed/dog456/512/512.jpg" \
    "${IMAGES_DIR}/test-dog.jpg" \
    "Dog image"

# Test image 3: Animal (cat)
download_file \
    "https://picsum.photos/seed/cat789/512/512.jpg" \
    "${IMAGES_DIR}/test-cat.jpg" \
    "Cat image"

# Test image 4: Scene (office)
download_file \
    "https://picsum.photos/seed/office101/512/512.jpg" \
    "${IMAGES_DIR}/test-office.jpg" \
    "Office scene"

# Test image 5: Scene (beach)
download_file \
    "https://picsum.photos/seed/beach202/512/512.jpg" \
    "${IMAGES_DIR}/test-beach.jpg" \
    "Beach scene"

# Additional test images for batch testing
for i in {1..10}; do
    download_file \
        "https://picsum.photos/seed/batch${i}/256/256.jpg" \
        "${IMAGES_DIR}/test-batch-${i}.jpg" \
        "Batch image ${i}"
done

echo "✓ All test images downloaded"

# Create test documents
echo ""
echo "Creating test documents..."
echo "-------------------------------------------"

# Simple text document
cat > "${DOCUMENTS_DIR}/test-document.txt" << 'EOF'
This is a test document about machine learning.

Machine learning is a subset of artificial intelligence (AI) that provides systems
the ability to automatically learn and improve from experience without being
explicitly programmed. Machine learning focuses on the development of computer
programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples,
direct experience, or instruction, in order to look for patterns in data and
make better decisions in the future based on the examples that we provide.

Key concepts in machine learning:
1. Supervised Learning - Learning with labeled training data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through interaction with environment
4. Deep Learning - Using neural networks with many layers

Machine learning has applications in:
- Computer vision and image recognition
- Natural language processing
- Speech recognition
- Recommendation systems
- Autonomous vehicles
- Medical diagnosis

The primary aim is to allow computers to learn automatically without human
intervention or assistance and adjust actions accordingly.
EOF

echo "✓ Test document created"

# Create QA test dataset
echo ""
echo "Creating QA test dataset..."
echo "-------------------------------------------"

cat > "${DATASETS_DIR}/visual-qa-test.json" << 'EOF'
{
  "dataset": "Visual QA Test Set",
  "version": "1.0",
  "description": "Test dataset for visual question answering",
  "images": [
    {
      "id": "img_001",
      "filename": "test-car.jpg",
      "questions": [
        {
          "question": "What color is the car?",
          "answer": "The car appears to be in the image"
        },
        {
          "question": "What type of vehicle is this?",
          "answer": "This is a car"
        },
        {
          "question": "Where is the vehicle located?",
          "answer": "The vehicle is in the image"
        }
      ]
    },
    {
      "id": "img_002",
      "filename": "test-dog.jpg",
      "questions": [
        {
          "question": "What animal is in the image?",
          "answer": "A dog"
        },
        {
          "question": "What is the dog doing?",
          "answer": "The dog is in the image"
        },
        {
          "question": "What color is the dog?",
          "answer": "The dog has fur"
        }
      ]
    },
    {
      "id": "img_003",
      "filename": "test-office.jpg",
      "questions": [
        {
          "question": "What objects are in this image?",
          "answer": "Office furniture and equipment"
        },
        {
          "question": "Is this an indoor or outdoor scene?",
          "answer": "This is an indoor scene"
        },
        {
          "question": "What room is this?",
          "answer": "This appears to be an office"
        }
      ]
    }
  ]
}
EOF

echo "✓ QA test dataset created"

# Create classification test dataset
echo ""
echo "Creating classification test dataset..."
echo "-------------------------------------------"

cat > "${DATASETS_DIR}/image-classification-test.json" << 'EOF'
{
  "dataset": "Image Classification Test Set",
  "version": "1.0",
  "description": "Test dataset for image classification",
  "classes": [
    {
      "name": "animal",
      "images": ["test-dog.jpg", "test-cat.jpg"]
    },
    {
      "name": "vehicle",
      "images": ["test-car.jpg"]
    },
    {
      "name": "scene",
      "images": ["test-office.jpg", "test-beach.jpg"]
    }
  ],
  "expected_accuracy": 0.75
}
EOF

echo "✓ Classification test dataset created"

# Create captioning test dataset
echo ""
echo "Creating captioning test dataset..."
echo "-------------------------------------------"

cat > "${DATASETS_DIR}/image-captioning-test.json" << 'EOF'
{
  "dataset": "Image Captioning Test Set",
  "version": "1.0",
  "description": "Test dataset for image captioning evaluation",
  "images": [
    {
      "filename": "test-car.jpg",
      "reference_captions": [
        "A car in the image",
        "A vehicle is shown",
        "This is a picture of a car"
      ]
    },
    {
      "filename": "test-dog.jpg",
      "reference_captions": [
        "A dog in the image",
        "A pet animal is shown",
        "This is a picture of a dog"
      ]
    },
    {
      "filename": "test-office.jpg",
      "reference_captions": [
        "An office interior",
        "A workspace with furniture",
        "This is a picture of an office"
      ]
    },
    {
      "filename": "test-beach.jpg",
      "reference_captions": [
        "A beach scene",
        "Coastal landscape",
        "This is a picture of a beach"
      ]
    }
  ],
  "evaluation_metrics": ["BLEU", "METEOR", "CIDEr"]
}
EOF

echo "✓ Captioning test dataset created"

# Create embedding similarity test dataset
echo ""
echo "Creating embedding similarity test dataset..."
echo "-------------------------------------------"

cat > "${DATASETS_DIR}/embedding-similarity-test.json" << 'EOF'
{
  "dataset": "Embedding Similarity Test Set",
  "version": "1.0",
  "description": "Test dataset for embedding similarity validation",
  "similar_pairs": [
    {
      "image1": "test-dog.jpg",
      "image2": "test-cat.jpg",
      "expected_similarity": 0.6,
      "reason": "Both are animals/pets"
    },
    {
      "image1": "test-office.jpg",
      "image2": "test-beach.jpg",
      "expected_similarity": 0.3,
      "reason": "Different scene types"
    }
  ],
  "dissimilar_pairs": [
    {
      "image1": "test-car.jpg",
      "image2": "test-dog.jpg",
      "expected_max_similarity": 0.4,
      "reason": "Vehicle vs animal"
    }
  ]
}
EOF

echo "✓ Embedding similarity test dataset created"

# Create README
echo ""
echo "Creating dataset README..."
echo "-------------------------------------------"

cat > "${DATA_DIR}/README.md" << 'EOF'
# Multimodal Test Data

This directory contains test data for multimodal inference testing.

## Directory Structure

```
test-resources/
├── images/           # Test images
│   ├── test-car.jpg
│   ├── test-dog.jpg
│   ├── test-cat.jpg
│   ├── test-office.jpg
│   ├── test-beach.jpg
│   └── test-batch-*.jpg
├── documents/        # Test documents
│   └── test-document.txt
└── datasets/         # Test datasets
    ├── visual-qa-test.json
    ├── image-classification-test.json
    ├── image-captioning-test.json
    └── embedding-similarity-test.json
```

## Usage

### Running Tests with Test Data

```bash
# Run integration tests
mvn test -Dtest="*IntegrationTest"

# Run with specific test data
mvn test -Dtest=GGUFMultimodalProcessorIntegrationTest#testLlavaVisualQA
```

### Dataset Descriptions

**Visual QA Test Set** (`visual-qa-test.json`)
- 3 images with QA pairs
- Tests visual question answering capabilities
- Expected accuracy: >80%

**Image Classification Test** (`image-classification-test.json`)
- 5 images across 3 classes
- Tests image classification
- Expected accuracy: >75%

**Image Captioning Test** (`image-captioning-test.json`)
- 4 images with reference captions
- Tests image captioning
- Evaluation metrics: BLEU, METEOR, CIDEr

**Embedding Similarity Test** (`embedding-similarity-test.json`)
- Similar and dissimilar image pairs
- Tests embedding quality
- Expected similarity thresholds defined

## Downloading Test Data

To download test images:

```bash
./download-test-data.sh
```

This will download sample images from Picsum Photos.

## Creating Custom Test Data

To use your own test images:

1. Place images in `images/` directory
2. Update dataset JSON files with your image filenames
3. Add reference captions/QA pairs as needed
4. Run tests

## License

Test images are from Picsum Photos (CC0 licensed).
Test datasets are created for testing purposes.
EOF

echo "✓ Dataset README created"

# Summary
echo ""
echo "========================================="
echo "✓ Test Data Download Complete!"
echo "========================================="
echo ""
echo "Downloaded/Created:"
echo "  - $(ls -1 ${IMAGES_DIR}/*.jpg 2>/dev/null | wc -l | tr -d ' ') test images"
echo "  - $(ls -1 ${DOCUMENTS_DIR}/*.* 2>/dev/null | wc -l | tr -d ' ') test documents"
echo "  - $(ls -1 ${DATASETS_DIR}/*.json 2>/dev/null | wc -l | tr -d ' ') test datasets"
echo ""
echo "Directories:"
echo "  - Images: ${IMAGES_DIR}"
echo "  - Documents: ${DOCUMENTS_DIR}"
echo "  - Datasets: ${DATASETS_DIR}"
echo ""
echo "Next steps:"
echo "  1. Review downloaded images"
echo "  2. Customize datasets if needed"
echo "  3. Run tests: mvn test"
echo ""
