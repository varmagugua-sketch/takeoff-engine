 """Generates a synthetic 36x24in test sheet: python make_test_pdf.py"""
import random
import fitz

random.seed(7)
doc = fitz.open()
page = doc.new_page(width=36 * 72, height=24 * 72)

page.draw_rect(fitz.Rect(36, 36, 36 * 72 - 36, 24 * 72 - 36), width=2)           # border
page.draw_rect(fitz.Rect(36 * 72 - 400, 24 * 72 - 300, 36 * 72 - 36, 24 * 72 - 36), width=1.5)  # title block
page.insert_text((36 * 72 - 380, 24 * 72 - 250), "TEST SHEET E-101", fontsize=24)

for _ in range(160):
    x = random.uniform(120, 36 * 72 - 450)
    y = random.uniform(120, 24 * 72 - 120)
    if random.random() < 0.5:
        page.draw_circle((x, y), 6, width=1)
    else:
        page.draw_rect(fitz.Rect(x - 6, y - 6, x + 6, y + 6), width=1)

doc.save("test_blueprint.pdf")
print("Wrote test_blueprint.pdf (36x24in, 160 symbols)")
