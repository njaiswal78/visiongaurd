# Visitor ID – Known Faces

Add photos of people you want the camera to recognize. Use the **filename** as the name (e.g. `rahul.jpg` → "rahul").

## Setup

1. **Install face_recognition:**
   ```bash
   pip install face_recognition
   ```

2. **Add face photos:**
   - One clear face per image
   - Front-facing works best
   - JPG format
   - Filename = display name (e.g. `rahul.jpg`, `mom.jpg`, `delivery_guy.jpg`)

## Examples

```
known_faces/
  rahul.jpg      → "Visitor: rahul"
  mom.jpg        → "Visitor: mom"
  john_doe.jpg   → "Visitor: john_doe"
```

When movement is detected or you use `/who` or `/detect`, recognized faces will show their name on the image and in the caption.
