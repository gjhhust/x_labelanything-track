import os
import base64
import contextlib
import io
import json
import os.path as osp

import PIL.Image

from ...app_info import __version__
from . import utils
from .logger import logger
from .label_converter import LabelConverter

PIL.Image.MAX_IMAGE_PIXELS = None
import PIL.Image
import numpy as np
import cv2
import io
import os.path as osp
import colorsys

@contextlib.contextmanager
def io_open(name, mode):
    assert mode in ["r", "w"]
    encoding = "utf-8"
    yield io.open(name, mode, encoding=encoding)


class LabelFileError(Exception):
    pass


class LabelFile:
    suffix = ".json"

    def __init__(self, filename=None):
        self.shapes = []
        self.image_path = None
        self.image_data = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: %s", filename)
            return None

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                image_pil = image_pil.convert("RGB")
                img_format = "JPEG"
            else:
                img_format = "PNG"
            image_pil.save(f, format=img_format)
            f.seek(0)
            return f.read()
        
    def load_image_file_by_cocoann(filename, ann_list, coco_category, get_color):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: %s", filename)
            return None

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        # Convert PIL Image to NumPy array
        image_np = np.array(image_pil.convert('RGB'))

        # Draw bounding boxes and category labels on the image
        for ann in ann_list:
            bbox = ann["bbox"]
            category_id = ann["category_id"]
            category_info = next((cat for cat in coco_category if cat["id"] == category_id), None)

            if category_info is not None:
                category_name = category_info["name"]
                track_id = ann.get("track_id", -1)

                # Get distinct color based on track_id
                r,g,b = get_color(category_name,int(track_id.split("_")[-1]))
                thickness = 2

                # Convert COCO bbox format (x, y, width, height) to OpenCV format (x1, y1, x2, y2)
                x, y, w, h = bbox
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                
                # Draw bounding box
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (int(r),int(g),int(b)), thickness)

                # Draw category label with smaller font size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_size = 0.8
                cv2.putText(image_np, category_name, (x1, y1 - 10), font, font_size, (int(r),int(g),int(b)), thickness)

        # Convert the annotated image back to bytes
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_format = "JPEG" if osp.splitext(filename)[1].lower() in [".jpg", ".jpeg"] else "PNG"
        _, image_bytes = cv2.imencode(f".{image_format.lower()}", image_np)
        return image_bytes.tobytes()

    def load_image_file_by_track_dict(filename, ann_list, all_labels, get_color):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: %s", filename)
            return None

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        # Convert PIL Image to NumPy array
        image_np = np.array(image_pil.convert('RGB'))

        # Draw bounding boxes and category labels on the image
        for ann in ann_list:
            bbox = ann["bbox"]
            label = ann["label"]
            group_id = ann["group_id"]

            if label in all_labels:
                label_index = all_labels.index(label)
                track_id = f"{label}_{group_id}"

                # Get distinct color based on track_id
                r,g,b = get_color(label,group_id)
                thickness = 2

                # Convert COCO bbox format (x, y, width, height) to OpenCV format (x1, y1, x2, y2)
                x, y, x2, y2 = bbox
                x1, y1, x2, y2 = int(x), int(y), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (int(r),int(g),int(b)), thickness)

                # Draw category label with smaller font size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_size = 0.8
                cv2.putText(image_np, label, (x1, y1 - 10), font, font_size, (int(r),int(g),int(b)), thickness)

        # Convert the annotated image back to bytes
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_format = "JPEG" if osp.splitext(filename)[1].lower() in [".jpg", ".jpeg"] else "PNG"
        _, image_bytes = cv2.imencode(f".{image_format.lower()}", image_np)
        return image_bytes.tobytes()
    

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        shape_keys = [
            "label",
            "points",
            "group_id",
            "difficult",
            "shape_type",
            "flags",
            "description",
            "attributes",
        ]
        try:
            with io_open(filename, "r") as f:
                data = json.load(f)
            version = data.get("version")
            if version is None:
                logger.warning(
                    "Loading JSON file (%s) of unknown version", filename
                )

            # Deprecated
            if data["shapes"]:
                for i in range(len(data["shapes"])):
                    shape_type = data["shapes"][i]["shape_type"]
                    shape_points = data["shapes"][i]["points"]
                    if shape_type == "rectangle" and len(shape_points) == 2:
                        logger.warning(
                            "UserWarning: Diagonal vertex mode is deprecated in X-AnyLabeling release v2.2.0 or later.\n"
                            "Please update your code to accommodate the new four-point mode."
                        )
                        data["shapes"][i][
                            "points"
                        ] = utils.rectangle_from_diagonal(shape_points)

            if data["imageData"] is not None:
                image_data = base64.b64decode(data["imageData"])
            else:
                # relative path from label file to relative path from cwd
                image_path = osp.join(osp.dirname(filename), data["imagePath"])
                image_data = self.load_image_file(image_path)
            flags = data.get("flags") or {}
            image_path = data["imagePath"]
            self._check_image_height_and_width(
                base64.b64encode(image_data).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes = [
                {
                    "label": s["label"],
                    "points": s["points"],
                    "shape_type": s.get("shape_type", "polygon"),
                    "flags": s.get("flags", {}),
                    "group_id": s.get("group_id"),
                    "description": s.get("description"),
                    "difficult": s.get("difficult", False),
                    "attributes": s.get("attributes", {}),
                    "other_data": {
                        k: v for k, v in s.items() if k not in shape_keys
                    },
                }
                for s in data["shapes"]
            ]
            for i, s in enumerate(data["shapes"]):
                if s.get("shape_type", "polygon") == "rotation":
                    shapes[i]["direction"] = s.get("direction", 0)
        except Exception as e:  # noqa
            raise LabelFileError(e) from e

        other_data = {}
        for key, value in data.items():
            if key not in keys:
                other_data[key] = value

        # Add new fields if not available
        other_data["text"] = other_data.get("text", "")

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.image_path = image_path
        self.image_data = image_data
        self.filename = filename
        self.other_data = other_data

    @staticmethod
    def _check_image_height_and_width(image_data, image_height, image_width):
        img_arr = utils.img_b64_to_arr(image_data)
        if image_height is not None and img_arr.shape[0] != image_height:
            logger.error(
                "image_height does not match with image_data or image_path, "
                "so getting image_height from actual image."
            )
            image_height = img_arr.shape[0]
        if image_width is not None and img_arr.shape[1] != image_width:
            logger.error(
                "image_width does not match with image_data or image_path, "
                "so getting image_width from actual image."
            )
            image_width = img_arr.shape[1]
        return image_height, image_width

    def save(
        self,
        filename=None,
        shapes=None,
        image_path=None,
        image_height=None,
        image_width=None,
        image_data=None,
        other_data=None,
        flags=None,
    ):
        if image_data is not None:
            image_data = base64.b64encode(image_data).decode("utf-8")
            image_height, image_width = self._check_image_height_and_width(
                image_data, image_height, image_width
            )

        if other_data is None:
            other_data = {}
        if flags is None:
            flags = {}
        data = {
            "version": __version__,
            "flags": flags,
            "shapes": shapes,
            "imagePath": image_path,
            "imageData": image_data,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }

        for key, value in other_data.items():
            assert key not in data
            data[key] = value
        try:
            with io_open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:  # noqa
            raise LabelFileError(e) from e

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
