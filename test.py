from canvas import Canvas

def main():
    canvas = Canvas(800, 600)
    canvas.draw_stroke(100, 200, 700, 500, 10, (255, 0, 0))  # Draw a red stroke
    canvas.draw_stroke(200, 500, 600, 300, 30, (128, 255, 0))  # Draw a green stroke
    canvas.save("output.png")

if __name__ == "__main__":
    main()
