import json
import os
import random
from PIL import Image

class AnnotationGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def generate_annotation(self, image_name, objects):
        annotations = {
            'image_name': image_name,
            'objects': objects
        }
        json_file = os.path.join(self.output_dir, f'{image_name}.json')
        with open(json_file, 'w') as f:
            json.dump(annotations, f, indent=4)

    def generate_annotations_for_images(self, image_names, num_objects_per_image):
        for image_name in image_names:
            objects = []
            for _ in range(num_objects_per_image):
                obj = {
                    'label': random.choice(['cat', 'dog', 'car', 'bicycle']),
                    'bbox': [random.randint(0, 640), random.randint(0, 480), random.randint(50, 100), random.randint(50, 100)]
                }
                objects.append(obj)
            self.generate_annotation(image_name, objects)

def generate_dummy_dataset(output_dir, num_images):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_images):
        image_name = f'image_{i}.png'
        image_path = os.path.join(output_dir, image_name)
        image = Image.new('RGB', (640, 480), color = (73, 109, 137))
        image.save(image_path)

    annotation_generator = AnnotationGenerator(output_dir)
    annotation_generator.generate_annotations_for_images([f'image_{i}.png' for i in range(num_images)], num_objects_per_image=3)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate a dummy dataset. 
    Optional - Specify the number of images to create.')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to generate')
    args = parser.parse_args()
    generate_dummy_dataset('dataset', args.num_images)