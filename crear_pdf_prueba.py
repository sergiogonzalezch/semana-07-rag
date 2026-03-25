# crear_pdf_prueba.py
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)

texto = """Promociones especiales de Cafe Aurora - Marzo 2026.

Lunes de Cortado: Todos los lunes, el cortado doble tiene 20% de descuento.
Precio normal: $48 MXN. Con descuento: $38 MXN.

Jueves de Postre: Al comprar cualquier bebida caliente, el pan de elote
artesanal tiene 50% de descuento. Precio normal: $35 MXN. Con descuento: $17.50 MXN.

Happy Hour: De 4:00 PM a 6:00 PM de lunes a viernes, todas las bebidas
frias tienen 30% de descuento.

Programa de Lealtad Aurora: Por cada 10 bebidas compradas, la numero 11
es gratis. Aplica para cualquier bebida del menu regular.
No acumulable con otras promociones."""

for linea in texto.strip().split("\n"):
    pdf.cell(0, 8, linea.strip(), new_x="LMARGIN", new_y="NEXT")

pdf.output("docs/promociones_marzo_2026.pdf")
print("PDF creado: docs/promociones_marzo_2026.pdf")