#===========================================================
#为网页前端的结果管理模块中的打印病历单pdf文件编写，将会在app.py中调用到
#===========================================================
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm
#========================================
#中文字体
#========================================
try:
    pdfmetrics.registerFont(TTFont('SimKai', 'C:/Windows/Fonts/simkai.ttf'))
    FONT_NAME = 'SimKai'
except:
    FONT_NAME = 'Helvetica'
def generate_pdf_report(patient_info, proba, threshold, risk_level, suggestions):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_path = temp_file.name
    temp_file.close()
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    # ==========标题==========
    c.setFont(FONT_NAME, 20)
    c.drawString(30 * mm, height - 30 * mm, "基于智能模型检测并导入的高血压风险评估报告")
    c.setFont(FONT_NAME, 12)
    c.drawString(30 * mm, height - 45 * mm, f"报告生成时间：{datetime.now().strftime('%Y年%m月%d日')}")
    # ==========分隔线==========
    c.line(30 * mm, height - 50 * mm, width - 30 * mm, height - 50 * mm)
    # ==========患者基本信息==========
    y = height - 55 * mm
    c.setFont(FONT_NAME, 14)
    c.drawString(30 * mm, y, "患者基本信息：")
    c.setFont(FONT_NAME, 12)
    y -= 8 * mm
    c.drawString(35 * mm, y, f"姓名：{patient_info.get('name', '未填写')}")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"身份证号：{patient_info.get('id_card', '未填写')}")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"就诊日期：{patient_info.get('visit_date', '')} {patient_info.get('visit_time', '')}")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"年龄：{patient_info.get('age', '')} 岁    性别：{patient_info.get('sex', '')}")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"BMI：{patient_info.get('bmi', '')}")
    # ==========评估结果==========
    y -= 10 * mm
    c.setFont(FONT_NAME, 14)
    c.drawString(30 * mm, y, "评估结果：")
    c.setFont(FONT_NAME, 12)
    y -= 10 * mm
    c.drawString(35 * mm, y, f"高血压风险概率：{proba:.1%}")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"筛查阈值：{threshold:.0%}")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"风险评估等级：{risk_level}")
    # ==========健康建议==========
    y -= 10 * mm
    c.setFont(FONT_NAME, 14)
    c.drawString(30 * mm, y, "健康建议：")
    c.setFont(FONT_NAME, 12)
    y -= 10 * mm
    for line in suggestions.split('\n'):
        if line.strip():
            c.drawString(35 * mm, y, line.strip())
            y -= 7 * mm
    #===========参考指标==========
    y -= 10 * mm
    c.setFont(FONT_NAME, 14)
    c.drawString(30 * mm, y, "各项指标评估参考：")
    c.setFont(FONT_NAME, 12)
    y -= 8 * mm
    c.drawString(35 * mm, y, f"血压正常值：收缩压<120mm/Hg、舒张压<80mm/Hg")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"血压正常高值：收缩压在120-129mm/Hg、舒张压<80mm/Hg")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"高血压一级：收缩压在130-139mm/Hg、舒张压在80-89mm/Hg")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"血压正常值：收缩压≥140mm/Hg、舒张压≥90mm/Hg")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"参考：2017 ACC/AHA 指南")
    # ===========签字位置==========
    y -= 10 * mm
    c.setFont(FONT_NAME, 14)
    c.drawString(30 * mm, y, "医生复核：")
    c.setFont(FONT_NAME, 12)
    y -= 8 * mm
    c.drawString(35 * mm, y, f"医生签名：_______________    日期：____年____月____日")
    y -= 8 * mm
    c.drawString(35 * mm, y, f"复核意见：□ 同意报告结论  □ 需进一步检查  □ 其他：_________")
    # ==========免责声明==========
    c.setFont(FONT_NAME, 10)
    c.drawString(30 * mm, 20 * mm, "※ 免责声明：本报告由机器学习算法辅助生成，仅供参考，不能替代专业医疗诊断。")
    c.save()
    return pdf_path