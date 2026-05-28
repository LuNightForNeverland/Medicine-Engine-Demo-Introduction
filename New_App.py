#=========================================================================================================
#基于streamlit的网页制作，包括模型导入，Deepseek API智能助手接入，诊断参数展示，管理诊断结果，模型参数配置和README简易版展示
#=========================================================================================================
import streamlit as st
import pandas as pd
import joblib as job
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

#==========创建预测历史记录==========
HISTORY_PATH = 'patient_history.csv'
def save_prediction(patient_data, proba, threshold, risk_level):
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'visit_date': patient_data.get('visit_date', ''),
        'visit_time': patient_data.get('visit_time', ''),
        'age': patient_data['Age'],
        'sex': '男' if patient_data['Sex_Male'] == 1 else '女',
        'bmi': patient_data['BMI'],
        'probability': round(proba, 4),
        'threshold': threshold,
        'risk_level': risk_level
    }
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(record)
    return True
def load_history():
    if 'history' in st.session_state:
        return pd.DataFrame(st.session_state.history)
    return pd.DataFrame()
#==========最佳阈值 已经在模型中计算获得==========
Best_Threshold = 0.30
#==========模型加载==========
@st.cache_resource
def load_model():
    model_path = 'Fusion_Model.pkl'
    return job.load(model_path)
model = load_model()
feature_cols = [
        'Age', 'BMI', 'Alcohol_Use', 'Physical_Activity', 'Diet_Salt_Intake',
        'Stress_Level', 'Heart_Rate_bpm',
        'Sex_Male', 'Diabetes_Yes', 'Family_Hx_Hypertension_Yes', 'Smoking_Status'
    ]
#==========标题和页面设置==========
st.set_page_config(
    page_title="高血压风险评估系统",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state = "expanded"
)
st.markdown("""
<style>
    div[data-testid="stImage"] {
        margin-top: -60px;
        margin-bottom: -50px;
    }
</style>
""", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col1:
    st.image('logo_of_gmu.png', use_container_width=True)
st.divider()
st.title("🏥高血压风险评估系统")
st.markdown("""
    本平台基于XGboost算法模型训练，支持医生输入患者信息、症状特征、检查数据等，
    通过人工智能算法辅助完成疾病诊断和风险评估。
    请在侧边栏填写信息，模型将会为您评估患者高血压风险以辅助后续治疗。
""")
#==========侧边栏设置，用于用户输入信息==========
with st.sidebar:
    st.markdown(f"""
        <div style="text-align: center; padding: 10px; background: #f0f2f6; border-radius: 10px;">
            <p style="margin:0; font-size:1.0rem; color: #666;">🕝️当前时间</p>
            <p style="margin:0; font-size:1.5rem; font-weight:600;">{datetime.now().strftime('%Y年%m月%d日 %H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)
    st.header("💻️患者基础信息")

    visit_date = st.date_input(
        "📅就诊日期",
        value=datetime.today()
    )

    time_options = [
        "08:00", "08:30", "09:00", "09:30", "10:00",
        "10:30", "11:00", "11:30", "12:00", "14:00",
        "14:30", "15:00", "15:30", "16:00", "16:30"
    ]

    visit_time = st.selectbox(
        "🕒️就诊时间",
        options= time_options,
        index= 2
    )

    st.subheader("🚶 患者身份")
    patient_name = st.text_input("姓名", placeholder="None")

    patient_id = st.text_input("病历号/身份证", placeholder="None")

    patient_age = st.number_input("患者年龄", min_value=18, max_value=100, value=18, step=1, help="患者年龄不低于18且不高于100")

    patient_sex = st.radio("性别", options=["女", "男"], horizontal=True)

    patient_bmi = st.number_input("BMI",min_value=10, max_value=50, value=24, step=1, help="患者BMI不低于10且不高于50")

    patient_alcohol = st.selectbox("患者饮酒情况", options=[-1, 0, 1, 2, 3],
                                    format_func=lambda x:["未知","从不","偶尔","经常","频繁"][x+1])

    patient_physical = st.selectbox("运动频率", options=[0, 1, 2, 3, 4],
                            format_func=lambda x: ["从不", "很少", "偶尔", "经常", "每天"][x])

    patient_stress = st.selectbox("压力水平", options=[0, 1, 2, 3],
                          format_func=lambda x: ["无", "轻度", "中度", "重度"][x])

    patient_heart_rate = st.slider("静息心率", 40, 200, 75)

    patient_diabetes = st.radio("是否有糖尿病", options=["否", "是"], horizontal=True)

    patient_family = st.radio("家族高血压史", options=["否", "是"], horizontal=True)

    patient_smoking = st.selectbox("吸烟状态", options=[0, 1, 2],
                           format_func=lambda x: ["从不吸烟", "曾吸烟", "当前吸烟"][x])

    patient_salt = st.selectbox("盐摄入量", options=[0, 1, 2],
                                format_func=lambda x: ["低盐", "正常", "高盐"][x])

    st.subheader("🩺 血压测量值（参考）")
    systolic = st.number_input("收缩压 (mmHg)", 70, 250, 120)
    diastolic = st.number_input("舒张压 (mmHg)", 40, 150, 80)

    st.caption("🔒 生产环境启用 AES-256 加密存储")
#==========中间内容分页==========
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊发病率预测", "📈诊断参数", "📋️结果管理", "📃模型参数配置", "📜README"])
# ==========页面1编辑==========
with tab1:
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        predict_clicked = st.button("🔍️开始评估", type="primary", use_container_width=True)
    # ==========预测逻辑处理==========
    if predict_clicked:
        # ==========整理输入数据==========
        input_data = {
            'Age': patient_age,
            'BMI': patient_bmi,
            'Alcohol_Use': patient_alcohol,
            'Physical_Activity': patient_physical,
            'Diet_Salt_Intake': patient_salt,
            'Stress_Level': patient_stress,
            'Sex_Male': 1 if patient_sex == "男" else 0,
            'Diabetes_Yes': 1 if patient_diabetes == "是" else 0,
            'Family_Hx_Hypertension_Yes': 1 if patient_family == "是" else 0,
            'Smoking_Status': patient_smoking,
            'visit_date': visit_date.strftime('%Y-%m-%d'),
            'visit_time': visit_time
        }

        # ==========转DataFrame==========
        input_df = pd.DataFrame([input_data])

        # ==========构造交互特征==========
        input_df['Age_BMI'] = input_df['Age'] * input_df['BMI']
        input_df['Age_FamilyHx'] = input_df['Age'] * input_df['Family_Hx_Hypertension_Yes']
        input_df['BMI_Diabetes'] = input_df['BMI'] * input_df['Diabetes_Yes']
        input_df['Smoking_Alcohol'] = input_df['Smoking_Status'] * input_df['Alcohol_Use']

        # ==========直接从模型获取期望的特征顺序==========
        if isinstance(model, dict):
            expected_features = model['xgb_model'].get_booster().feature_names
        else:
            expected_features = model.get_booster().feature_names
        # ==========按模型期望的顺序排列==========
        input_df = input_df[expected_features]
        # ==========预测
        if isinstance(model, dict):
            # ==========融合模型：用四个基模型与元模型预测==========
            xgb_proba = model['xgb_model'].predict_proba(input_df)[:, 1][0]
            cat_proba = model['cat_model'].predict_proba(input_df)[:, 1][0]
            rf_proba = model['rf_model'].predict_proba(input_df)[:, 1][0]
            lr_proba = model['lr_model'].predict_proba(input_df)[:, 1][0]

            X_meta = np.array([[xgb_proba, cat_proba, rf_proba, lr_proba]])
            proba = model['meta_model'].predict_proba(X_meta)[:, 1][0]
        else:
            # ==========单模型直接预测==========
            proba = model.predict_proba(input_df)[:, 1][0]
        # ==========保存到 session_state==========
        st.session_state.prediction_done = True
        st.session_state.last_proba = proba
        st.session_state.last_input = input_data
        st.session_state.patient_context = f"年龄{patient_age}岁，BMI{patient_bmi}，高血压风险概率{proba:.1%}"
        # ==========清空之前的聊天记录==========
        if "messages" in st.session_state:
            del st.session_state.messages
        # ==========刷新页面以显示结果==========
        st.rerun()

    # ==========页面显示预测结果==========
    if st.session_state.get("prediction_done", False):
        proba = st.session_state.last_proba
        st.subheader("💡预测结果")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="高血压风险概率", value=f"{proba:.1%}")
            if proba < Best_Threshold:
                st.success(f"✅ 低风险（阈值 {Best_Threshold:.0%}）")
            else:
                st.error(f"🔴 高风险（阈值 {Best_Threshold:.0%}）")
        with col2:
            st.markdown("#### 📋 建议")
            if proba < Best_Threshold:
                st.markdown("- 保持健康生活方式😊\n- 定期体检🕒️\n- 控制盐分摄入🧂")
            else:
                st.markdown("- 尽快就医检查🏥\n- 测量实际血压❤️\n- 改善生活习惯🤒")
                st.warning("⚠️本结果仅供参考，不能替代专业医疗诊断。")
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.info("👈️请在侧边栏填写患者信息，然后点击「开始评估」")
#==========页面2编辑==========
with tab2:
    st.markdown("### 📄高血压诊断参考标准")
    col1, col2 = st.columns(2)
    #==========页面2\栏1==========
    with col1:
        st.markdown("""
        #### 📋️血压分级标准
        | 类别 | 收缩压 | 舒张压 |
        |------|--------|--------|
        | 正常 | < 120 | < 80 |
        | 正常高值 | 120-129 | < 80 |
        | 高血压1级 | 130-139 | 80-89 |
        | 高血压2级 | ≥ 140 | ≥ 90 |
        *参考：2017 ACC/AHA 指南*
        """)
    #==========页面2\栏2==========
    with col2:
        st.markdown("""
        #### 主要危险因素
        - 🧓 **年龄**：男性 ≥ 45岁，女性 ≥ 55岁
        - 📊 **BMI**：≥ 24 为超重，≥ 28 为肥胖
        - 🧂 **高盐饮食**：每日盐摄入 > 6g
        - 🍺 **过量饮酒**：每周 > 14 标准杯
        - 🚬 **吸烟**：当前吸烟或曾吸烟
        - 🩺 **糖尿病**：空腹血糖 ≥ 7.0 mmol/L
        - 👨‍👩‍👦 **家族史**：一级亲属有高血压
        """)
    st.divider()
    st.markdown("### 🩺 生活方式建议")
    col3, col4, col5 = st.columns(3)
    # ==========页面2\栏3==========
    with col3:
        st.info("🥗 **饮食建议**\n\n- 控制盐摄入 < 6g/天\n- 多吃蔬菜水果\n- 减少饱和脂肪")
    # ==========页面2\栏4==========
    with col4:
        st.success("🏋️ **运动建议**\n\n- 每周 ≥ 150分钟中强度运动\n- 快走、游泳、力量训练\n- 保持健康体重")
    # ==========页面2\栏5==========
    with col5:
        st.warning("🚫 **避免事项**\n\n- 戒烟限酒\n- 减轻精神压力\n- 保证充足睡眠")
#==========页面3编辑==========
with tab3:
    st.markdown("### 📊患者评估记录")
    col1, col2, col3 = st.columns(3)
    #==========页面3栏1==========
    with col1:
        if st.button("📁保存本次评估", use_container_width=True):
            if 'last_input' in st.session_state:
                risk = "高风险" if st.session_state.last_proba >= Best_Threshold else "低风险"
                save_prediction(
                    st.session_state.last_input,
                    st.session_state.last_proba,
                    Best_Threshold,
                    risk
                )
                st.success("评估结果已保存")
            else:
                st.warning("请先进行一次评估")
    # ==========页面3栏2==========
    with col2:
        from pdf_of_app import generate_pdf_report
        if st.button("📁生成PDF报告", use_container_width=True):
            if st.session_state.get("prediction_done",False):
                proba = st.session_state.last_proba
                risk = "高风险" if proba >= Best_Threshold else "低风险"
                patient_info = {
                    'name': patient_name,
                    'id_card': patient_id,
                    'visit_date': visit_date.strftime('%Y-%m-%d'),
                    'visit_time': visit_time,
                    'age': patient_age,
                    'sex': patient_sex,
                    'bmi': f"{patient_bmi:.1f}"
                }
                suggestions = "• 保持健康生活方式\n• 定期体检\n• 控制盐分摄入 < 6g/天\n• 每周运动 ≥ 150 分钟" if proba < Best_Threshold else "• 尽快就医进行血压测量\n• 建议做 24 小时动态血压监测\n• 控制盐摄入 < 5g/天\n• 遵医嘱，不要自行用药"
                pdf_path = generate_pdf_report(patient_info, proba, Best_Threshold, risk, suggestions)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇️ 下载 PDF 报告",
                        data=f,
                        file_name=f"高血压评估报告_{patient_name}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
            else:
                st.warning("请先进行一次评估")
    st.divider()
    if 'history' in st.session_state and st.session_state.history:
        st.subheader("预测历史")
        for record in st.session_state.history[::-1]:  # 倒序显示，最新的在前
            st.write(f"时间: {record['timestamp']}, 风险: {record['risk_level']}, 概率: {record['probability']:.4f}")
        # ==========导出功能==========
        df_history = pd.DataFrame(st.session_state.history)
        csv = df_history.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="🖨️导出全部记录 (CSV)",
            data=csv,
            file_name=f"高血压评估记录_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("😉暂无历史评估记录，请先进行评估并保存。")
    #==========页脚==========
    st.markdown("---")
    st.markdown("© 当前演示版本为方便展示数据结构，采用本地 CSV 明文存储。实际生产环境中，系统将采用 AES-256 算法对姓名、身份证号进行字段级加密存储，并启用数据库透明加密，确保患者隐私安全。")
    #==========页面4==========
    with tab4:
        #==========特征重要性图==========
        st.markdown("### 🧩特征重要性图")
        st.divider()
        img_path1 = '2_SHAP_Bar.png'
        st.image(img_path1 ,width = 600)
        st.markdown("""
        #### 🥇 核心主导特征
        - **Age_BMI（年龄与BMI交互）**：SHAP 值高达 0.45，在所有特征中排名第一。这表明年龄与肥胖的协同放大效应是模型决策的最核心依据。
        - **Alcohol_Use（饮酒）**：SHAP 值 0.42，排名第二，是独立风险因素中贡献最大的。
        - **Family_Hx_Hypertension_Yes（家族史）**：SHAP 值 0.40，排名第三，遗传背景对高血压风险有显著影响。
        #### 🥈 次级关键特征
        - **Age_FamilyHx（年龄与家族史交互）**：SHAP 值 0.39，排名第四，年龄越大且有家族史，风险叠加效应明显。
        - **Age（年龄）**：SHAP 值 0.35，排名第五，年龄本身仍是重要的独立预测因子。
        - **Physical_Activity（运动频率）**：SHAP 值 0.32，排名第六，缺乏运动显著增加高血压风险。
        - **Smoking_Status（吸烟状态）**：SHAP 值 0.31，排名第七，吸烟是重要的可干预危险因素。
        #### 🥉 辅助特征
        - **Diet_Salt_Intake（盐摄入）**：SHAP 值 0.25，高盐饮食对血压有直接影响。
        - **BMI_Diabetes（BMI与糖尿病交互）**：SHAP 值 0.24，肥胖合并糖尿病进一步放大风险。
        - **BMI（体重指数）**：SHAP 值 0.22，肥胖本身是高血压的独立危险因素。
        - **Sex_Male（男性）**：SHAP 值 0.18，男性相对风险更高。
        - **Diabetes_Yes（糖尿病）**：SHAP 值 0.16，糖尿病与高血压高度共病。
        - **Stress_Level（压力水平）**：SHAP 值 0.05，贡献相对较小。
        - **Smoking_Alcohol（吸烟与饮酒交互）**：SHAP 值 0.02，贡献最小。
        #### 📌 结论
        - 本文构造的**四项交互特征全部进入 SHAP 重要性前十二**，其中 Age_BMI 高居榜首，Age_FamilyHx 位列第四，BMI_Diabetes 位列第十。充分验证了**领域知识驱动的特征工程**对模型预测能力的关键作用。
        - 模型特征重要性排序与临床医学认知高度一致，表明模型已有效学习到年龄、生活方式、遗传因素及其交互效应对高血压患病风险的影响权重。
        """)
        st.divider()
        #==========混淆矩阵热力图==========
        st.markdown("### 🧩混淆矩阵热力图")
        st.divider()
        img_path2 = '2_Confusion_Matrix.png'
        st.image(img_path2, width = 600)
        st.markdown("""
            #### 总体分类概况
            - 该混淆矩阵展示了模型在测试数据集上的分类结果分布。矩阵对角线元素（220 和 762）数值显著高于非对角线元素，表明模型在两个类别上均实现了较高比例的准确分类。
            #### 类别识别详情
            - **高血压类别（标签1）**：模型正确识别 762 例，漏诊 100 例。召回率（灵敏度）为 **88.4%**，表明模型能够有效捕捉高血压患者的特征模式，漏诊率较低。
            - **正常类别（标签0）**：模型正确识别 220 例，误判 231 例。精确率为 **48.8%**，表明模型倾向于将正常样本误判为高血压。
            #### 误差分布特征
            - **假阳性（误诊）**：231 例正常样本被误判为高血压。
            - **假阴性（漏诊）**：100 例高血压患者被漏判。
            - 模型采取了**高敏感度策略**，优先确保对潜在高血压病例的检出，代价是一定程度的误诊率。这在疾病筛查场景中是可接受的权衡。
            #### 综合指标
            - 准确率：74.8%
            - 精确率（高血压）：76.8%
            - 召回率（高血压）：88.4%
            - F1 分数（高血压）：0.82
            """)
        st.divider()
        #==========AUC图==========
        st.markdown("### 🧩AUC图")
        st.divider()
        img_path3 = '3_ROC_Curve.png'
        st.image(img_path3, width=600)
        st.markdown("""
            #### 曲线形态分析
            - 图示 ROC 曲线展示了模型在不同分类阈值下的性能表现。曲线显著向左上方凸起，远离对角线参考线（虚线），表明模型在维持较低假阳性率的同时，能够获得较高的真阳性率。
            #### 曲线下面积 (AUC)
            - 图例显示，该 ROC 曲线下的面积 **AUC = 0.79**。该数值定量地反映了模型的整体区分能力。根据通用评估标准，AUC 在 0.7-0.8 之间表示模型具有**中上等的预测效能**，能够以 79% 的概率正确区分随机抽取的高血压患者与非患者。
            #### 阈值权衡特性
            - 曲线斜率在低假阳性率区间（横轴 0.0 至 0.2）较陡，说明在严格控制误报率的前提下，模型仍能保持较高的检出率。随着假阳性率的增加，曲线逐渐趋于平缓，表明单纯提高阈值对提升检出率的边际效益递减。
            #### 结论
            - AUC = 0.79 表明融合模型在高血压风险预测任务上具有良好的区分能力，能够为临床辅助决策提供有价值的参考。
            """)
        st.divider()
        #==========F1图==========
        st.markdown("### 🧩F1图")
        st.divider()
        img_path4 = '04_F1_Score.png'
        st.image(img_path4, width=600)
        st.markdown("""
        #### 🏆 总体排名
        - **SVM**：F1 = 0.7864，排名第一，在精确率与召回率的平衡上表现最佳。
        - **LightGBM**：F1 = 0.7844，排名第二，Boosting 模型中表现最优。
        - **Fusion_Model**：F1 = 0.7821，排名第三，异质集成展现出良好的分类平衡能力。
        - **XGBoost**：F1 = 0.7806，排名第四，与融合模型差距仅 0.0015。
        - **CATBoost**：F1 = 0.7798，排名第五。
        - **RandomForest**：F1 = 0.7790，排名第六。
        - **Logistic**：F1 = 0.7570，排名第七，线性模型在分类平衡上相对弱势。
        #### 📊 分层解读
        ##### 🥇 第一梯队（F1 ≥ 0.78）
        - **SVM、LightGBM、Fusion_Model、XGBoost、CATBoost、RandomForest** 六个模型的 F1 分数集中在 0.7790-0.7864 之间，差距极小（最大仅 0.0074）。
        - 这表明树模型及其集成在高血压风险预测任务上具有**稳定且相近的分类平衡能力**。
        ##### 🥈 第二梯队（F1 < 0.76）
        - **逻辑回归** F1 分数为 0.7570，与第一梯队差距约 0.02-0.03。
        - 尽管逻辑回归在 AUC 上表现优异，但其 F1 分数相对较低，说明在默认阈值下，线性模型在精确率与召回率的平衡上略逊于非线性模型。
        #### 📌 关键发现
        | 模型 | AUC 排名 | F1 排名 | 综合表现 |
        |:---|:---:|:---:|:---|
        | SVM | 7 | **1** | AUC 最低，F1 最高（特殊现象） |
        | LightGBM | 5 | **2** | 分类平衡能力突出 |
        | **Fusion_Model** | **2** | **3** | **AUC 与 F1 双高，综合最优** |
        | 逻辑回归 | **1** | 7 | AUC 夺冠，F1 垫底 |
        #### 🔍 SVM 异常现象分析
        - SVM 在 AUC（0.7304）上排名垫底，但在 F1 分数（0.7864）上排名第一。
        - 这表明 SVM 在默认阈值（0.5）下具有较好的分类平衡能力，但其整体排序能力（AUC）较弱。
        - **AUC 评估排序能力，F1 评估分类平衡能力，二者不可互相替代。**
        #### 📌 结论
        - **Fusion_Model 是唯一在 AUC（排名第2）和 F1（排名第3）上均位列前三的模型**，展现出最优的综合性能。
        - 逻辑回归虽在 AUC 上夺冠，但其 F1 分数较低，提示在样本不均衡场景下，AUC 与 F1 需结合评估。
        - 本实验充分验证了**多指标评估体系**的必要性：单一指标可能掩盖模型在不同维度的优劣。
        """)
        st.divider()
        #==========核心指标==========
        st.markdown("### 📃核心指标")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("测试集 AUC", "0.790")
        with col2:
            st.metric("准确率", "75%")
        with col3:
            st.metric("召回率(患者)", "88%")
        with col4:
            st.metric("特征数量", "14")
        st.divider()
        #==========训练日志可视化==========
        st.markdown("### 📈训练日志可视化曲线")
        st.divider()
        log_path = 'training_log.txt'
        auc_values = []
        with open(log_path, "r") as f:
            for line in f:
                match = re.search('validation_0-auc:([\d.]+)', line)
                if match:
                    auc_values.append(float(match.group(1)))
        if auc_values:
            fig, ax = plt.subplots(figsize = (10, 4))
            ax.plot(auc_values,color = "skyblue", linewidth = 2)
            ax.set_xlabel("Training Rounds")
            ax.set_ylabel("Validation AUC")
            ax.set_title(f"Model convergence curve(Best: {max(auc_values): .4f}), Final:{auc_values[-1]:.4f}")

            best_idx = auc_values.index(max(auc_values))
            ax.scatter(best_idx, max(auc_values), color='grey', s=50, zorder=5)
            ax.annotate(f'Best: {max(auc_values):.4f}',
                        xy=(best_idx, max(auc_values)),
                        xytext=(best_idx + 5, max(auc_values) - 0.01),
                        fontsize=10)
            st.pyplot(fig)
            st.markdown("""
            <style>
                div1[data-testid="stImage"] {
                    margin-top: 20px;
                    margin-bottom: 20px;
                }
            </style>
            """, unsafe_allow_html=True)
            with st.expander("📄 查看原始训练日志", expanded=False):
                with open(log_path, "r", encoding="utf-8") as f:
                    st.text(f.read())
    with tab5:
        # ==========核心亮点==========
        st.markdown("### 🏆 核心亮点")
        st.markdown("""
        - **白名单防泄露**：剔除诊断指标，AUC从0.99回归至0.79
        - **8模型对比**：单模+三融+四融
        - **SHAP可解释性**：Age_BMI贡献度第一
        """)
        # ==========代码块用 st.code()==========
        st.markdown("#### 配置环境所需要的安装包")
        st.code("pip install -r requirements.txt", language="bash")
        # ==========图片用 st.image()==========
        st.divider()
        st.markdown("#### 8模型ROC曲线对比")
        st.divider()
        img_path = 'All_Models_ROC.png'
        st.image(img_path, width=600)
        # ==========表格==========
        st.divider()
        st.markdown("#### 模型性能汇总")
        df = pd.DataFrame({
            '模型': ['XGBoost', 'LightGBM', 'CatBoost', '四融合'],
            '测试AUC': [0.7862, 0.7833, 0.7909, 0.7900],
            'F1分数': [0.7806, 0.7844, 0.7798, 0.7821],
            '过拟合差距': [0.0897, 0.0891, 0.0542, 0.0394]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
