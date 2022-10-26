'''
A PDF report generator. 
See details at https://pyfpdf.readthedocs.io/en/latest/index.html'''
from fpdf import FPDF
from datetime import datetime

paper_length, paper_height = 210, 297
class PDF(FPDF):
    my_header = None
    def header(self):# Overwrite page header: use it to set date
        self.set_font('Arial', 'I', 10)
        self.cell(8)
        self.cell(30, 5, str(datetime.now()), 0, 0, 'C')
        self.ln(10)
    def footer(self):# Overwrite page footer
        self.set_y(-15) # Position at 1.5 cm from bottom
        self.set_font('Arial', 'I', 8)  # Arial italic 8
        self.set_text_color(128)    # Text color in gray
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + ' / {nb}', 0, 0, 'C')
    def increment_x(self, x):
        self.set_x(self.get_x()+x)
    def increment_y(self, y):
        self.set_y(self.get_y()+y)
    def get_pos(self):
        return self.get_x(), self.get_y()
    def print_title(self, title):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 0, 'C')
        self.ln(15)
    def print_cell(self, w, h, content):
        pass
    def print_author(self, author):
        self.set_font('Courier', '', 12)
        self.cell(0, 5, author, 0, 0, 'C')
        self.ln(15)
    def print_section(self, section):
        self.set_font('Arial', '', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, section, 0, 0, 'L', 1)
        self.ln(12)
    def print_paragraph(self, content, tail_vspace=3):
        self.set_font('Times', '', 11) 
        self.multi_cell(0, 5, content)  # Output justified text
        self.ln(tail_vspace)  # Line break
    def print_image(self, path, w=90, h=50, space='full', caption=None): 
        if self.get_y() + w > paper_height: # page break if the fig does not fit
            self.add_page()
        if space == 'full':
            self.image(path, x=10, y=self.get_y(), w=w, h=h)
            if caption is not None:
                self.set_font('Times', 'I', 10)
                self.text(self.get_x()+10, pdf.get_y()+h+2, caption)
            self.increment_y(h+10)
        elif space == 'left_half': # Must be followed by another image
            self.image(path, x=self.get_x(), y=self.get_y(), w=w, h=h)
            if caption is not None:
                self.set_font('Times', 'I', 10)
                self.text(self.get_x()+10, self.get_y()+h+2, caption)
        elif space == 'right_half':
            self.image(path, x=paper_length/2, w=w, h=h)
            if caption is not None:
                self.set_font('Times', 'I', 10)
                self.text(10+paper_length/2, self.get_y()+2, caption)
            self.increment_y(10)
        else:
            print("Set space to 'full' for standalone image, 'left_half' and 'right_half' for images side by side")
    def __smart_row(self, w, h, n_cols, data):
        x, y = self.get_pos()
        next_y = 0
        for i in range(n_cols): 
            self.multi_cell(w, h, data[i]) 
            x += w
            next_y = max(next_y, self.get_y())
            self.set_xy(x, y)
        self.set_xy(10, next_y)
    def print_table(self, data, header=None, caption=None, cell_width=10, cell_height=10): # data = num_rows x num_cols
        n = len(data[0])
        cell_width = (paper_length-10*2)/n
        if header is not None:
            self.set_font('Courier', 'B', 11)
            self.__smart_row(cell_width, cell_height, n, header)
            self.set_line_width(0.5)
            self.line(self.get_x(), self.get_y(), paper_length-10, self.get_y())
        self.set_font('Courier', '', 11)
        for row in data: 
            self.__smart_row(cell_width, cell_height, n, row)
        if caption is not None:
            self.set_font('Times', 'I', 10)
            self.cell(0, 5, caption, align='C')
            self.ln()
if __name__ == '__main__':
    pdf = PDF()
    pdf.add_page()
    pdf.print_title('PDF Report Generation Demo')
    pdf.print_author('Comp Fab Group')
    pdf.print_section('Section 1 title')
    pdf.print_paragraph('This is a' + 50*' long'+' paragraph.\nnew line here'+'\nnew line')
    pdf.print_paragraph('Another paragraph.')
    pdf.print_section('Section 2 title')

    pdf.print_image('chart.png', space='full', caption='Fig 1')

    pdf.print_paragraph('after figure '*20)
    pdf.print_image('chart.png', space='left_half', caption='Subfig 1')
    pdf.print_image('chart.png', space='right_half', caption='Subfig 2')
    #pdf.print_image('chart.png', space='half', caption='Subfig 2')
    pdf.print_paragraph('After a pair of subfigures...')
    pdf.alias_nb_pages() # Record total page numbers
    pdf.add_page()

    pdf.print_paragraph("Another page")
    pdf.line(pdf.get_x(), pdf.get_y(), paper_length-10, pdf.get_y())
    pdf.ln(5) # vertical space
    pdf.dashed_line(pdf.get_x(), pdf.get_y(), paper_length-10, pdf.get_y(), 1, 1)
    pdf.print_paragraph("Two lines above.")
    # Draw a rectangle in the center
    pdf.line(paper_length/2-10, pdf.get_y(), paper_length/2+10, pdf.get_y())
    pdf.line(paper_length/2+10, pdf.get_y(), paper_length/2+10, pdf.get_y()+10)
    pdf.line(paper_length/2+10, pdf.get_y()+10, paper_length/2-10, pdf.get_y()+10)
    pdf.line(paper_length/2-10, pdf.get_y()+10, paper_length/2-10, pdf.get_y())
    pdf.increment_y(20)
    # Print a table
    #header = [f"Model {i}" for i in range(1, 4)]
    header = ['A', 'B'*30, "C"]
    data = [['a'*12*i*j for j in range(1, 4)] for i in range (1, 3)]
    data.append(['More']*3)
    pdf.print_table(data=data, header=header, caption='Table Title')

    pdf.print_paragraph('More content goes here.')

    pdf.output('fpdf_pdf_report.pdf', 'F')