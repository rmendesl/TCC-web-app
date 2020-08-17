import pandas as pd 
import numpy as np
import streamlit as st
from pycaret.classification import *
import base64
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)

model = load_model('modelrematricula_pycaret')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions_df = predictions_df['Label'][0]
    return predicitons

def movecol(df, cols_to_move=[], ref_col='', place='After'):
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3])

def EDA(df):
    # Preenche os valores das variáveiS ausentes com 0 (zeros)
    df['VAL_A_PAGAR'] = df['VAL_A_PAGAR'].fillna(0)
    df['VAL_A_PAGAR_PAR'] = df['VAL_A_PAGAR_PAR'].fillna(0)
    df['VAL_DIVIDA_MENS'] = df['VAL_DIVIDA_MENS'].fillna(0)
    df['VAL_DIVIDA_TOTAL'] = df['VAL_DIVIDA_TOTAL'].fillna(0)
    df['CR_PER_ANT'] = df['CR_PER_ANT'].fillna(0)
    df['QTD_ACESSOS_19_2'] = df['QTD_ACESSOS_19_2'].fillna(0)
    df['QTD_ACESSOS_20_1'] = df['QTD_ACESSOS_20_1'].fillna(0)

    # Preenche os valores das variáveis com o um valor padrão de acordo com a situação adequada
    df['RISCO_INADIMPLENCIA'] = df['RISCO_INADIMPLENCIA'].fillna('SEM RISCO')
    df['FAIXA_DE_DIVIDA'] = df['FAIXA_DE_DIVIDA'].fillna('SEM DIVIDA')
    df['PERDA_FINANCIAMENTO'] = df['PERDA_FINANCIAMENTO'].fillna('')

    return df.copy()


def run():
    image = Image.open('Logo.png')
    st.image(image, use_column_width=True, use_column_height=True)
    st.title('Instituição de Ensino Superior XY')
    st.subheader('Esse app irá realizar a previsão de rematrículas dos alunos.')
    #st.sidebar.info('Esse app irá realizar a previsão de rematrículas dos alunos.')
    
    #st.sidebar.success('https://www.pycaret.org')
    #st.sidebar.header('User Input Features')  

    file_upload = st.file_uploader("Faça o upload do seu arquivo XLSX para as previsões.", type=["xlsx"])

    if file_upload is not None:
        data = pd.read_excel(file_upload, sheet_name=0)
        df_clean = EDA(data)
       
        # Realizando as previsões
        predictions = predict_model(estimator=model, data=df_clean)
        predictions['Rematriculado'] = predictions['Label'].apply(lambda x: 'SIM' if x == 1 else 'NÃO' )

        predictions = movecol(predictions, cols_to_move=['Label', 'Rematriculado', 'Score'],  ref_col='COD_MATRICULA')

        # Atributos para serem exibidos por padrão
        defaultcols = ['COD_MATRICULA', 'Label', 'Score', 'Rematriculado']

        # Defindo atributos a partir do multiselect
        cols = st.multiselect("Atributos", predictions.columns.tolist(), default=defaultcols)
        st.dataframe(predictions[cols])

        if st.checkbox("Salvar resultados"):  
            df_save = predictions[cols]
            csv = df_save.to_csv(encoding='utf-8-sig', index=False)
            b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download arquivo CSV</a> (clique com o botão direito e Salvar link como &lt;algum_nome&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)

        if st.checkbox("JSON Format"):        
            df_list = pd.DataFrame(predictions[cols].values , columns =  cols)
            df_list = df_list.to_json(orient='records', force_ascii=False)
            st.json(df_list)
            b64 = base64.b64encode(df_list.encode('utf-8-sig')).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a href="data:file/json;base64,{b64}" download="results.json">Download arquivo JSON</a> (clique com o botão direito e Salvar link como &lt;algum_nome&gt;.json)'
            st.markdown(href, unsafe_allow_html=True)
            
if __name__ == '__main__':
    run()