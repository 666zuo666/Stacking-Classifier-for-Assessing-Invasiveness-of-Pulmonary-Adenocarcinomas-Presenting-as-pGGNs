import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

# 加载模型
model_path = "stacking_classifier_model.pkl"
stacking_regressor = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="Stacking Classifier with SHAP for Model Interpretation ", page_icon="📊")

st.title("📊 Stacking Classifier with SHAP for Model Interpretation ")
st.write("""
Performing model predictions based on input feature values, and understanding the contribution of features to model predictions through SHAP analysis results.。
""")

# 左侧侧边栏输入区域
st.sidebar.header("Feature Input Panel")
st.sidebar.write("Features：")

# 定义特征输入范围
Radscore = st.sidebar.number_input( "Radscore:",min_value=-0.5, max_value=0.6, value=-0.251002483713187)
Size = st.number_input("Tumor size:", min_value=0.0, max_value=30.0, value=7.62, step=0.01)
CT_value = st.number_input("Mean CT value:", min_value=-800.0, max_value=0.0, value=-480.66, step=1.0)
Age = st.number_input("Age:", min_value=21.0, max_value=100.0, value=64.0, step=1.0)
# 添加预测按钮 
predict_button = st.sidebar.button("Predict")

# 主页面用于结果展示
if predict_button:
    st.header("Prediction result")
    try:
        # 将输入特征转换为模型所需格式
        input_array = np.array(Radscore,Size,CT_value,Age)
        #input_array = np.array(Radscore,Size,CT_value,Age).reshape(1, -1)

        # 模型预测
        prediction = stacking_regressor.predict(input_array)[0]

        # 显示预测结果
        st.success(f"Prediction result：{prediction:.2f}")
    except Exception as e:
        st.error(f"error：{e}")

# 可视化展示
st.header("SHAP")
st.write("""
The following figures present the SHAP analysis results of the model, including the feature contributions of the first-layer base learners, the second-layer meta-learner, and the entire stacking model.
""")

# 第一层基学习器 SHAP 可视化
st.subheader("1.  first-layer base learners")
st.write("Feature contribution analysis of the base learners (RF, XGBoost, LightGBM, GBDT, AdaBoost, CatBoost).")
first_layer_img = "summary_plot.png"
try:
    img1 = Image.open(first_layer_img)
    st.image(img1, caption="Feature contribution analysis of the base learners.", use_column_width=True)
except FileNotFoundError:
    st.warning("error")

# 第二层元学习器 SHAP 可视化
st.subheader("2. second-layer meta-learner")
st.write("Feature contribution analysis of second-layer meta-learner.")
meta_layer_img = "SHAP Contribution Analysis for the Meta-Learner in the Second Layer.png"
try:
    img2 = Image.open(meta_layer_img)
    st.image(img2, caption="SHAP Contribution Analysis for the Meta-Learner in the Second Layer ", use_column_width=True)
except FileNotFoundError:
    st.warning("error")

# 整体 Stacking 模型 SHAP 可视化
st.subheader("3.  Stacking classfier")
st.write("Feature contribution analysis of the stacking Classfier")
overall_img = "Based on the overall feature contribution analysis of SHAP to the stacking model.png"
try:
    img3 = Image.open(overall_img)
    st.image(img3, caption="Stacking classfier", use_column_width=True)
except FileNotFoundError:
    st.warning("error")

# 页脚
st.markdown("---")
# st.header("总结")
# st.write("""
# 通过本页面，您可以：
# 1. 使用输入特征值进行实时预测。
# 2. 直观地理解第一层基学习器、第二层元学习器以及整体 Stacking 模型的特征贡献情况。
# 这些分析有助于深入理解模型的预测逻辑和特征的重要性。
# """)
