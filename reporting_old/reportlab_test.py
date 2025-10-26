import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

# 1. Carica il CSV con pandas
df = pd.read_csv("/Users/daniele_leuteri/PycharmProjects/tracking_perf_nsat/e2e-sp-refactored/exports/target_exports/OUTPUT_CSV/HGV_330.csv")

# 2. Prepara i dati per ReportLab: 
#    una lista di liste, con intestazioni + righe
data = [df.columns.tolist()] + df.values.tolist()

# 3. Crea il documento
doc = SimpleDocTemplate("output.pdf", pagesize=letter)

# 4. Crea la tabella
table = Table(data)

# 5. Applica uno stile (opzionale ma consigliato)
style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
])
table.setStyle(style)

# 6. Costruisci il PDF
elements = [table]
doc.build(elements)