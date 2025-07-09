from qreader import QReader
from PIL import Image
import numpy as np
import binascii # For robust hex decoding error handling
import base64
import json
import zlib
import cv2
import io

# --- Configuración ---
CHARUCO_CONFIG = {
    'SQUARES_X': 5,
    'SQUARES_Y': 5,
    'SQUARE_LENGTH_MM': 10.0,
    'MARKER_LENGTH_MM': 7.0,
    'DICTIONARY_NAME': "DICT_4X4_100"
}


def cv_image_to_base64(cv_image):
    """Converts an OpenCV image (numpy array) to a base64 encoded string.

    This is used to embed image data directly into HTML or JSON responses for
    display in a web browser. The image is converted from BGR (OpenCV's default)
    to RGB, then saved as a JPEG in-memory, and finally base64 encoded.

    Args:
        cv_image (numpy.ndarray): The input image in OpenCV format (BGR color).

    Returns:
        str | None: A data URI string (e.g., "data:image/jpeg;base64,...")
            representing the image, or None if the input `cv_image` is None.
    """
    print("Attempting to convert OpenCV image to base64.")
    if cv_image is None:
        print("Input OpenCV image is None, returning None.")
        return None
    
    # Convert BGR to RGB
    print("Converting BGR to RGB.")
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    print("Converted to PIL image.")
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    print("Successfully converted OpenCV image to base64 JPEG.")
    return f"data:image/jpeg;base64,{image_base64}"

def _decode_zlib_json_qr(qr_text_content):
    """
    Attempts to decode a QR text content assuming it's a hex-encoded,
    zlib-compressed JSON string.

    Args:
        qr_text_content (str): The raw string content from a QR code.

    Returns:
        dict or None: The decoded JSON object if successful, otherwise None.
    """
    if not isinstance(qr_text_content, str):
        return None
    try:
        compressed_data = bytes.fromhex(qr_text_content)
        json_str = zlib.decompress(compressed_data).decode('utf-8')
        decoded_data = json.loads(json_str)
        return decoded_data
    except (ValueError, binascii.Error, zlib.error, UnicodeDecodeError, json.JSONDecodeError):
        # ValueError for non-hex string in fromhex
        # binascii.Error for odd-length string or non-hex characters in fromhex
        # zlib.error for decompression issues
        # UnicodeDecodeError for utf-8 decoding issues
        # json.JSONDecodeError for JSON parsing issues
        # print(f"    Debug: Failed to decode/decompress QR content as zlib/JSON: {e}") # Optional
        return None

def detect_and_draw_qrcodes(image_input):
    """
    Reads an image from disk, detects QR codes in it,
    draws a green quadrilateral around each detected QR code, and attempts
    to decode zlib-compressed JSON content from the QR text.

    Args:
        image_input (str or numpy.ndarray): Path to the input image file or the image itself (as a NumPy array).
    Returns:
        tuple (list[numpy.ndarray], list[str], list[Optional[dict]]) or (None, None, None):
            - A list of images:
                - The first image is the input image with QR codes highlighted.
                  If no QR codes are found, it's the original unmodified image.
                - Subsequent images are cropped individual QR code regions.
            - A list of strings, where each string is the decoded text of a
              corresponding QR code. The order matches the cropped images.
            - A list of decoded JSON objects (dict) or None if decoding failed
              for the corresponding QR code text. The order matches the other lists.
            Returns (None, None, None) if the image cannot be read or if input type is invalid.
            Returns ([original_image], [], []) if no QR codes are found.
    """
    # Create a QReader instance
    qreader_detector = QReader()

    if isinstance(image_input, str):
        # Input is a path, load the image
        original_image = cv2.imread(image_input)
        if original_image is None:
            print(f"Error: Could not read image from path: '{image_input}'")
            return None, None, None
    elif isinstance(image_input, np.ndarray):
        original_image = image_input.copy() # Work on a copy
    else:
        print(f"Error: Invalid input type. Expected string path or NumPy array, got {type(image_input)}.")
        return None, None, None

    # Initialize image_for_display with the original. It will be copied if modifications are made.
    image_for_display = original_image 
    cropped_qr_images = []
    decoded_texts_list = []
    decoded_json_objects_list = []

    # QReader expects images in RGB format, OpenCV reads in BGR by default
    # Use the original_image for conversion, as it's pristine.
    rgb_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Step 1: Detect QR codes to get bounding boxes.
    # qreader.detect() returns a list of bounding boxes (numpy arrays of points),
    # or None if no QR codes are found.
    detected_bboxes = qreader_detector.detect(image=rgb_img)

    if detected_bboxes:  # Handles None or an empty list
        # Determine the source for logging
        image_source_name = image_input if isinstance(image_input, str) else "the provided image array"
        print(f"Found {len(detected_bboxes)} potential QR code(s) in {image_source_name}.")

        made_modifications_to_display_image = False

        for i, detection_info in enumerate(detected_bboxes):
            current_decoded_text = None
            try:
                # Step 2: Decode the text for each detected QR code using its bounding box.
                current_decoded_text = qreader_detector.decode(image=rgb_img, detection_result=detection_info)
            except Exception as e:
                print(f"  Error decoding potential QR Code #{i+1}: {e}. Detection info: {detection_info}.")
                continue # Skip to the next detection

            if current_decoded_text is not None:
                # Successfully decoded. Now check for corners to draw and crop.
                quad_corners = detection_info.get('quad_xy')

                if quad_corners is not None:
                    # This is a confirmed QR code with location.
                    if not made_modifications_to_display_image:
                        image_for_display = original_image.copy() # Copy before first drawing
                        made_modifications_to_display_image = True
                    
                    print(f"  QR Code #{i+1} decoded: '{current_decoded_text[:50]}{'...' if len(current_decoded_text) > 50 else ''}'")
                    try:
                        current_points = np.array(quad_corners, dtype=np.float32)
                        centroid = np.mean(current_points, axis=0)
                        expanded_points = centroid + 1.1 * (current_points - centroid)

                        img_height, img_width = original_image.shape[:2]
                        expanded_points[:, 0] = np.clip(expanded_points[:, 0], 0, img_width - 1)
                        expanded_points[:, 1] = np.clip(expanded_points[:, 1], 0, img_height - 1)

                        points_for_drawing = np.array(expanded_points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(image_for_display, [points_for_drawing], isClosed=True, color=(0, 255, 0), thickness=4)

                        # --- Crop the QR region from the original_image ---
                        x_coords = expanded_points[:, 0]
                        y_coords = expanded_points[:, 1]
                        crop_x_start = int(np.min(x_coords))
                        crop_y_start = int(np.min(y_coords))
                        crop_x_end = int(np.max(x_coords)) + 1
                        crop_y_end = int(np.max(y_coords)) + 1

                        if crop_x_start < crop_x_end and crop_y_start < crop_y_end:
                            cropped_qr_img = original_image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
                            if cropped_qr_img.size > 0:
                                cropped_qr_images.append(cropped_qr_img)
                                decoded_texts_list.append(current_decoded_text) # Add text IFF crop is successful
                                json_obj = _decode_zlib_json_qr(current_decoded_text)
                                decoded_json_objects_list.append(json_obj)
                            else:
                                print(f"  QR Code #{i+1} (decoded) resulted in an empty crop slice. Not adding to results.")
                        else:
                            print(f"  QR Code #{i+1} (decoded) has invalid dimensions for cropping. Not adding to results.")
                    except (ValueError, TypeError) as e:
                        print(f"  Error processing/drawing polygon for decoded QR Code #{i+1}: {e}. Quad corners: {quad_corners}. Not adding to results.")
                else:
                    # Decoded, but no quad_corners
                    print(f"  QR Code #{i+1} was decoded ('{current_decoded_text[:50]}...') but 'quad_xy' (corners) are missing. Cannot draw or crop.")
            else:
                # current_decoded_text is None: Detected by bbox, but not a decodable QR.
                print(f"  Potential QR Code #{i+1} was detected by bounding box, but could not be decoded. No box drawn.")
    else:
        # No bounding boxes detected at all
        image_source_name = image_input if isinstance(image_input, str) else "the provided image array"
        print(f"No QR codes found in {image_source_name}.")

    # If no modifications were made, image_for_display is still the original_image.
    # Otherwise, it's a copy with drawings.
    return [image_for_display] + cropped_qr_images, decoded_texts_list, decoded_json_objects_list

def detect_charuco_board(image_input, squares_x, squares_y, square_length_mm, marker_length_mm, dictionary_name, display=False):
    """
    Detects a ChArUco board in an image and draws the detected corners and board.

    Args:
        image_input (str or numpy.ndarray): Path to the input image or the image itself (as a NumPy array).
        squares_x (int): Number of squares in X direction of the board.
        squares_y (int): Number of squares in Y direction of the board.
        square_length_mm (float): Length of a square in millimeters.
        marker_length_mm (float): Length of a marker in millimeters.
        dictionary_name (str): Name of the Aruco dictionary used (e.g., "DICT_4X4_50").
        display (bool): Whether to display the image with detections.

    Note:
        - The function uses `cv2.aruco.CharucoDetector` for detection, which
        internally handles ArUco marker detection and ChArUco corner interpolation.
        - Physical lengths (`square_length_mm`, `marker_length_mm`) are converted
          to meters for `cv2.aruco.CharucoBoard` initialization.

    Returns:
        tuple or (None, None, None, None, None):
            - img (numpy.ndarray or None): The image with detections drawn. None if an error occurs (e.g., image not loaded).
            - charucoCorners (numpy.ndarray or None): Array of detected ChArUco corners. None if no corners are found or an error occurs.
            - charucoIds (numpy.ndarray or None): Array of IDs for the detected ChArUco corners. None if no corners are found or an error occurs.
            - markerCorners (list of numpy.ndarray or None): List of detected ArUco marker corners. None if no markers are found or an error occurs.
            - markerIds (numpy.ndarray or None): Array of IDs for the detected ArUco markers. None if no markers are found or an error occurs.
    """

    # Load the image
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            print(f"Error: Could not load image from path: {image_input}")
            return None, None, None, None, None
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy() # Work on a copy to avoid modifying the original array
    else:
        print("Error: Invalid image_input type. Must be a path (str) or a NumPy array.")
        return None, None, None, None, None

    # Get the ArUco dictionary
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    except AttributeError:
        print(f"Error: Dictionary '{dictionary_name}' not found. Please check the dictionary name.")
        return None, None, None, None, None

    # Create the ChArUco board object (same as in generation)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length_mm / 1000.0, marker_length_mm / 1000.0, dictionary)
    # board = cv2.aruco.CharucoBoard((squares_x, squares_y), 0.01, 0.007, dictionary)
    
    # Set legacy pattern for older ChArUco boards
    # board.setLegacyPattern(True)
    
    # --- NEW: Create CharucoParameters and DetectorParameters ---
    # DetectorParameters for the underlying Aruco detection
    detector_params = cv2.aruco.DetectorParameters()
    # CharucoParameters for the ChArUco interpolation/detection
    charuco_params = cv2.aruco.CharucoParameters()

    # --- NEW: Pass charuco_params and detector_params to CharucoDetector ---
    # The CharucoDetector constructor now expects (board, charucoParams, detectorParams)
    charucoDetector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)


    # Use detectBoard() to get charuco corners directly
    charucoCorners, charucoIds, markerCorners, markerIds = charucoDetector.detectBoard(img)

    if markerIds is not None:
        print(f"Detected {len(markerIds)} Aruco markers.")

        if charucoIds is not None:
            print(f"Detected {len(charucoIds)} ChArUco corners.")

            # Draw the detected ChArUco corners
            cv2.aruco.drawDetectedCornersCharuco(img, charucoCorners, charucoIds, cornerColor=(0, 255, 0))

            # Draw the individual ArUco markers (optional, as charuco detection is more robust)
            cv2.aruco.drawDetectedMarkers(img, markerCorners, markerIds,  borderColor=(0, 0, 255))

            # --- Pose Estimation (Optional, requires camera calibration) ---
            # If you have camera calibration parameters (camera_matrix, dist_coeffs),
            # you can estimate the pose of the board.
            # For example:
            # ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            #     charucoCorners, charucoIds, board, camera_matrix, dist_coeffs, None, None
            # )
            # if ret:
            #     print("Board pose estimated.")
            #     cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.05) # Draw axes on the board

        else:
            print("No ChArUco corners detected from the Aruco markers.")
    else:
        print("No Aruco markers detected in the image.")

    # Display the result
    if display:
        cv2.imshow("ChArUco Board Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img, charucoCorners, charucoIds, markerCorners, markerIds

def process_image(image_path):
    """Loads an image from a file path and processes it for ChArUco and QR codes.

    This function performs the core image analysis:
    1. Reads the image file using OpenCV.
    2. Converts the original image to base64 for display.
    3. Calls `detect_and_draw_qrcodes` to find and decode QR codes, drawing on a copy.
    4. Calls `detect_charuco_board` on the (potentially QR-annotated) image.
    5. Converts the final processed image to base64.

    Args:
        image_path (str): The local file system path to the image to be processed.

    Returns:
        dict: A dictionary containing the processing results:
            - 'original_image' (str): Base64 encoded original image.
            - 'processed_image' (str): Base64 encoded image with detections drawn.
            - 'charuco_detected' (bool): True if a ChArUco board was found.
            - 'qr_codes' (list[str]): A list of decoded string data from QR codes.
            - 'qr_codes_json' (list[dict]): A list of decoded JSON objects from QR codes.
            Returns a dictionary with default values if the image cannot be loaded.
    """
    print(f"Starting image processing for: {image_path}")
    result = {
        'original_image': None,
        'processed_image': None,
        'charuco_detected': False,
        'qr_codes': [],
        'qr_codes_json': []
    }
    
    # Load image
    print(f"Loading image from path: {image_path}")
    cv_image = cv2.imread(image_path)
    if cv_image is None:
        print(f"Failed to load image from path: {image_path}")
        return result
    
    print(f"Successfully loaded image: {image_path}")
    # Convert original image to base64
    result['original_image'] = cv_image_to_base64(cv_image)
    
    # Start with copy for processing
    processed_image = cv_image.copy()
    print("Created a copy of the image for processing.")
    
    # QR Code detection
    if detect_and_draw_qrcodes:
        print("Attempting QR code detection.")
        try:
            qr_images, qr_decoded_texts, qr_decoded_json_objects = detect_and_draw_qrcodes(cv_image)
            if qr_images and len(qr_images) > 0 and qr_images[0] is not None:
                processed_image = qr_images[0]
                print(f"QR code detection successful. Found {len(qr_decoded_texts)} QR codes.")
                if qr_decoded_texts:
                    result['qr_codes'] = qr_decoded_texts
                    result['qr_codes_json'] = qr_decoded_json_objects
            else:
                print("QR code detection ran, but no QR codes found or image not returned.")
        except Exception as e:
            print(f"Exception during QR code detection for {image_path}: {e}", exc_info=True)
    else:
        print("detect_and_draw_qrcodes module not available. Skipping QR detection.")

    # ChArUco detection
    if detect_charuco_board:
        try:
            charuco_output, charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco_board(
                processed_image,
                CHARUCO_CONFIG['SQUARES_X'], CHARUCO_CONFIG['SQUARES_Y'],
                CHARUCO_CONFIG['SQUARE_LENGTH_MM'], CHARUCO_CONFIG['MARKER_LENGTH_MM'],
                CHARUCO_CONFIG['DICTIONARY_NAME'], display=False
            )
            if charuco_output is not None:
                processed_image = charuco_output
                if charuco_ids is not None and len(charuco_ids) > 0:
                    print(f"ChArUco board detection successful. Found {len(charuco_ids)} ChArUco IDs.")
                    result['charuco_detected'] = True
                else:
                    print("ChArUco board detection ran, image updated, but no ChArUco IDs found.")
            else:
                print("ChArUco board detection ran but returned None.")
        except Exception as e:
            print(f"Exception during ChArUco board detection for {image_path}: {e}", exc_info=True)
    else:
        print("detect_charuco_board module not available. Skipping ChArUco detection.")

    result['processed_image'] = cv_image_to_base64(processed_image)
    print(f"Finished image processing for: {image_path}. Charuco detected: {result['charuco_detected']}, QR codes: {len(result['qr_codes'])}")
    return result
