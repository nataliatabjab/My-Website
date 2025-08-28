PImage img;
ArrayList<PVector> selectedPoints = new ArrayList<PVector>();
PVector offset = new PVector(0, 0);  // Panning offset
PVector prevMouse = new PVector(0, 0);
boolean isDragging = false;

// Scale factor to fit image within window while maintaining aspect ratio
float scaleFactor;

void settings() {
  img = loadImage("25mm.bmp");  // Load the image first
  
  // Determine scale factor to fit within screen while maintaining aspect ratio
  float wScale = (float) displayWidth / img.width;
  float hScale = (float) displayHeight / img.height;
  scaleFactor = min(wScale, hScale); // Choose the smaller scaling factor

  size((int)(img.width * scaleFactor), (int)(img.height * scaleFactor));  // Set canvas size
}

void setup() {
  image(img, offset.x, offset.y, img.width * scaleFactor, img.height * scaleFactor);
}

void draw() {
  background(0);
  
  // Draw scaled image
  image(img, offset.x, offset.y, img.width * scaleFactor, img.height * scaleFactor);

  // Draw selected points (convert back to scaled space)
  fill(255, 0, 0);
  for (PVector p : selectedPoints) {
    ellipse(p.x * scaleFactor + offset.x, p.y * scaleFactor + offset.y, 10, 10);
  }

  // If we have 2 points, calculate pixel distance in original image space
  if (selectedPoints.size() == 2) {
    float pixelDist = dist(selectedPoints.get(0).x, selectedPoints.get(0).y, 
                           selectedPoints.get(1).x, selectedPoints.get(1).y);
    
    float pixelSize = 2.2e-6;  // Camera pixel size in meters (2.2 µm)
    float physicalDist = pixelDist * pixelSize;  // Convert to real-world distance
    
    fill(255);
    textSize(16);
    text("Pixel Distance: " + pixelDist, 20, 40);
    text("Physical Distance: " + physicalDist + " m", 20, 60);
  }
}

void mousePressed() {
  if (mouseButton == LEFT) {
    // Convert mouse click to original image space
    float origX = (mouseX - offset.x) / scaleFactor;
    float origY = (mouseY - offset.y) / scaleFactor;

    if (selectedPoints.size() < 2) {
      selectedPoints.add(new PVector(origX, origY));
    } else {
      selectedPoints.clear();  // Reset selection
    }
  } else if (mouseButton == RIGHT) {
    isDragging = true;
    prevMouse.set(mouseX, mouseY);
  }
}

void mouseDragged() {
  if (isDragging) {
    float dx = mouseX - prevMouse.x;
    float dy = mouseY - prevMouse.y;
    offset.add(dx, dy);
    prevMouse.set(mouseX, mouseY);
  }
}

void mouseReleased() {
  isDragging = false;
}
