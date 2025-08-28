float x = 0.1;
float y = 0.0;
float z = 0.0;

float a = 0.12;
float b = 0.02;
float c = 13;

float dt = 0.01;
float scale = 10;
//float angle = 0;

float rotX = 0;
float rotY = 0;

ArrayList<PVector> points = new ArrayList<PVector>();

void setup() {
  size(800, 600, P3D);
  background(0);
  stroke(255);
  noFill();
}

void draw() {
  background(0);

  // Rotate the whole system before drawing
  translate(width/2, height/2, -200);
  rotateY(rotY);
  rotateX(rotY);
  
  // Evolve
  for (int i = 0; i < 10; i++) {
    float dx = (-y - z) * dt;
    float dy = (x + a * y) * dt;
    float dz = (b + z * (x - c)) * dt;

    x += dx;
    y += dy;
    z += dz;

    points.add(new PVector(x, y, z));
  }

  // Limit trail length for performance
  if (points.size() > 3000) {
    points.remove(0);
  }

  // Draw
  stroke(255);
  beginShape();
  for (PVector p : points) {
    vertex(p.x * scale, p.y * scale, p.z * scale);
  }
  endShape();

}


void mouseDragged() {
  float sensitivity = 0.01;
  float dx = (mouseX - pmouseX) * sensitivity;
  float dy = (mouseY - pmouseY) * sensitivity;

  rotY += dx;
  rotX += dy;
}
