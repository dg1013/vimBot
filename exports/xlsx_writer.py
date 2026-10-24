"""A small, dependency-free .xlsx writer (stdlib only: zipfile + XML strings).

Enough of the OOXML spreadsheet format for a data hand-off: many sheets,
typed cells (numbers, booleans, inline strings), a bold shaded header row
that stays frozen, an autofilter over every column, estimated column widths
and a few number formats. No formulas, no shared-string table (inline
strings are simpler and every reader accepts them), no charts.

Why not openpyxl: it is not in the project env, and `src/` is stdlib-only by
policy (CLAUDE.md §13); a 200-line writer keeps the export reproducible with
nothing to install.

Usage:
    wb = Workbook()
    wb.add_sheet("Corpus", header=[...], rows=[[...], ...],
                 formats={"exec_match_pct": "pct"})
    wb.save("out.xlsx")

Cell values: None -> empty; bool -> boolean cell; int/float -> number
(NaN/inf -> empty); anything else -> str(). Strings longer than Excel's
32,767-character cell limit are cut with a marker; characters XML 1.0 cannot
carry (control characters other than tab/newline/CR) are spelled out as
\\xNN so nothing is silently lost.
"""

import math
import re
import zipfile
from datetime import datetime, timezone

CELL_LIMIT = 32767
_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")
# columns of a percent format hold FRACTIONS (0.521), not 52.1
NUM_FORMATS = {"pct": 2, "float3": 3, "int": 4, "float1": 5}


def _esc(text):
    text = _ILLEGAL.sub(lambda m: "\\x%02x" % ord(m.group(0)), text)
    if len(text) > CELL_LIMIT:
        text = text[:CELL_LIMIT - 24] + " …[cut at 32767 chars]"
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;"))


def col_letter(n):
    """1 -> A, 27 -> AA."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _sheet_name(name, taken):
    bad = '[]:*?/\\'
    clean = "".join("-" if c in bad else c for c in name).strip("' ")[:31].strip() or "Sheet"
    base, i = clean, 2
    while clean.lower() in taken:
        suffix = " (%d)" % i
        clean = base[:31 - len(suffix)] + suffix
        i += 1
    taken.add(clean.lower())
    return clean


class Workbook:
    def __init__(self):
        self.sheets = []          # (name, xml, dimension)
        self._names = set()

    def add_sheet(self, name, header, rows, formats=None, widths=None,
                  tab_color=None, freeze=True, autofilter=True, max_width=60):
        """`formats` maps header name -> 'pct' | 'float3' | 'float1' | 'int'.
        `widths` maps header name -> character width (else estimated)."""
        name = _sheet_name(name, self._names)
        formats = formats or {}
        widths = widths or {}
        ncols = len(header)
        style_by_col = [NUM_FORMATS.get(formats.get(h), 0) for h in header]
        # estimate widths from the header and the first 400 rows
        est = [min(max_width, max(8, len(str(h)) + 2)) for h in header]
        for row in rows[:400]:
            for j, v in enumerate(row[:ncols]):
                if v is None:
                    continue
                s = str(v)
                first = s.split("\n", 1)[0]
                est[j] = max(est[j], min(max_width, len(first) + 2))
        parts = []
        parts.append('<row r="1">')
        for j, h in enumerate(header):
            parts.append('<c r="%s1" t="inlineStr" s="1"><is><t xml:space="preserve">%s</t></is></c>'
                         % (col_letter(j + 1), _esc(str(h))))
        parts.append("</row>")
        for i, row in enumerate(rows, start=2):
            cells = []
            for j, v in enumerate(row[:ncols]):
                if v is None:
                    continue
                ref = "%s%d" % (col_letter(j + 1), i)
                st = style_by_col[j]
                sattr = ' s="%d"' % st if st else ""
                if isinstance(v, bool):
                    cells.append('<c r="%s" t="b"%s><v>%d</v></c>' % (ref, sattr, int(v)))
                elif isinstance(v, (int, float)):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        continue
                    cells.append('<c r="%s"%s><v>%r</v></c>' % (ref, sattr, v))
                else:
                    cells.append('<c r="%s" t="inlineStr"%s><is><t xml:space="preserve">%s</t></is></c>'
                                 % (ref, sattr, _esc(str(v))))
            parts.append('<row r="%d">%s</row>' % (i, "".join(cells)))
        nrows = len(rows) + 1
        dim = "A1:%s%d" % (col_letter(max(ncols, 1)), nrows)
        cols = "".join('<col min="%d" max="%d" width="%s" customWidth="1"/>'
                       % (j + 1, j + 1, widths.get(h, est[j])) for j, h in enumerate(header))
        views = ('<sheetViews><sheetView workbookViewId="0"%s>'
                 % (' tabSelected="1"' if not self.sheets else ""))
        if freeze:
            views += ('<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
                      '<selection pane="bottomLeft" activeCell="A2" sqref="A2"/>')
        views += "</sheetView></sheetViews>"
        pr = '<sheetPr><tabColor rgb="FF%s"/></sheetPr>' % tab_color if tab_color else ""
        xml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
               'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
               '%s<dimension ref="%s"/>%s<sheetFormatPr defaultRowHeight="15"/>'
               '<cols>%s</cols><sheetData>%s</sheetData>%s</worksheet>'
               % (pr, dim, views, cols, "".join(parts),
                  '<autoFilter ref="%s"/>' % dim if autofilter and ncols else ""))
        self.sheets.append((name, xml, dim, autofilter and bool(ncols)))
        return name

    # ------------------------------------------------------------ package

    def save(self, path):
        ct = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
              '<Default Extension="xml" ContentType="application/xml"/>'
              '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
              '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
              '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
              '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>']
        wb = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
              'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
              '<bookViews><workbookView xWindow="0" yWindow="0" windowWidth="28800" windowHeight="16000"/></bookViews><sheets>']
        rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">']
        names = []
        for i, (name, _xml, dim, filt) in enumerate(self.sheets, start=1):
            ct.append('<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % i)
            wb.append('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (_esc(name), i, i))
            rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet%d.xml"/>' % (i, i))
            if filt:
                a, b = dim.split(":")
                ref = "$%s$%s:$%s$%s" % (re.match("[A-Z]+", a).group(0), a[len(re.match("[A-Z]+", a).group(0)):],
                                         re.match("[A-Z]+", b).group(0), b[len(re.match("[A-Z]+", b).group(0)):])
                names.append('<definedName name="_xlnm._FilterDatabase" localSheetId="%d" hidden="1">\'%s\'!%s</definedName>'
                             % (i - 1, _esc(name).replace("'", "''"), ref))
        ct.append("</Types>")
        wb.append("</sheets>")
        if names:
            wb.append("<definedNames>%s</definedNames>" % "".join(names))
        wb.append("</workbook>")
        n = len(self.sheets)
        rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>' % (n + 1))
        rels.append("</Relationships>")
        styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                  '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                  '<numFmts count="3"><numFmt numFmtId="164" formatCode="0.0%"/><numFmt numFmtId="165" formatCode="0.000"/><numFmt numFmtId="166" formatCode="0.0"/></numFmts>'
                  '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>'
                  '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
                  '<fill><patternFill patternType="solid"><fgColor rgb="FFDDEBF7"/><bgColor indexed="64"/></patternFill></fill></fills>'
                  '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
                  '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
                  '<cellXfs count="6">'
                  '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
                  '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
                  '<xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
                  '<xf numFmtId="165" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
                  '<xf numFmtId="1" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
                  '<xf numFmtId="166" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/>'
                  '</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
        root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                     '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
                     '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                     '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
                     '</Relationships>')
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
                'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
                'xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
                '<dc:creator>vimBot exports/build_workbook.py</dc:creator>'
                '<dcterms:created xsi:type="dcterms:W3CDTF">%s</dcterms:created>'
                '<dcterms:modified xsi:type="dcterms:W3CDTF">%s</dcterms:modified></cp:coreProperties>' % (now, now))
        app = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
               '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
               'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
               '<Application>vimBot xlsx_writer</Application></Properties>')
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("[Content_Types].xml", "".join(ct))
            z.writestr("_rels/.rels", root_rels)
            z.writestr("docProps/core.xml", core)
            z.writestr("docProps/app.xml", app)
            z.writestr("xl/workbook.xml", "".join(wb))
            z.writestr("xl/_rels/workbook.xml.rels", "".join(rels))
            z.writestr("xl/styles.xml", styles)
            for i, (_name, xml, _dim, _f) in enumerate(self.sheets, start=1):
                if i > 1:
                    xml = xml.replace(' tabSelected="1"', "", 1)
                z.writestr("xl/worksheets/sheet%d.xml" % i, xml)
