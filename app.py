import math
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title='GEO-RETO 360 | Taludes', page_icon='⛰️', layout='wide')

def terreno(x,H,ratio):
    x=np.asarray(x)
    crest=ratio*H
    return np.where(x<0,0.0,np.where(x<crest,x/ratio,H)).astype(float)

def calcular(H,ratio,gamma,c,phi,n,xe,cx,cy,ru):
    if H<=0 or ratio<=0 or gamma<=0 or xe<=ratio*H or c<0 or not 0<=phi<60 or not 0<=ru<1:
        raise ValueError('Revisa H, pendiente, γ, c, φ, ru y salida x (debe superar la coronación).')
    R=math.hypot(cx,cy)
    if abs((xe-cx)**2+(H-cy)**2-R**2)>0.01:
        raise ValueError('El círculo no pasa por la salida. Activa «Ajustar centro Y automáticamente» o corrige el centro.')
    if cy<=H or max(abs(cx),abs(xe-cx))>=R:
        raise ValueError('El círculo no define una base válida para toda la sección.')
    def base(x):
        x=np.asarray(x)
        return cy-np.sqrt(np.maximum(0,R**2-(x-cx)**2))
    edges=np.linspace(0,xe,n+1)
    xm=(edges[:-1]+edges[1:])/2
    areas=[]
    for a,b in zip(edges[:-1],edges[1:]):
        xx=np.linspace(a,b,21)
        thick=terreno(xx,H,ratio)-base(xx)
        if np.min(thick)<-1e-5:
            raise ValueError('La superficie circular sobresale del terreno. Cambia la geometría.')
        areas.append(float(np.trapezoid(thick,xx) if hasattr(np,'trapezoid') else np.trapz(thick,xx)))
    widths=np.diff(edges)
    thick=terreno(xm,H,ratio)-base(xm)
    W=gamma*np.asarray(areas)
    alpha=np.arctan((xm-cx)/np.sqrt(R**2-(xm-cx)**2))
    sn=np.sin(alpha); cs=np.cos(alpha)
    l=widths/cs
    u=ru*gamma*thick
    U=u*l
    t=math.tan(math.radians(phi))
    drive=float(np.sum(W*sn))
    if drive<=0: raise ValueError('Momento desestabilizador no positivo; revisa la geometría.')
    resisting=c*l+(W*cs-U)*t
    fell=float(np.sum(resisting)/drive)
    bishop=max(fell,0.2)
    for it in range(500):
        m=cs+sn*t/bishop
        if np.any(m<=0): raise ValueError('No es válida la iteración de Bishop con esta geometría.')
        nxt=float(np.sum((c*widths+(W-u*widths)*t)/m)/drive)
        if not np.isfinite(nxt) or nxt<=0: raise ValueError('Bishop no converge para estos parámetros.')
        if abs(nxt-bishop)<1e-9:
            bishop=nxt;break
        bishop=nxt
    else: raise ValueError('Bishop no convergió en 500 iteraciones.')
    df=pd.DataFrame({'Dovela':np.arange(1,n+1),'x centro (m)':xm,'Ancho (m)':widths,'Área (m²)':areas,
                     'Peso (kN/m)':W,'Ángulo base (°)':np.degrees(alpha),'Longitud base (m)':l,
                     'Altura suelo (m)':thick,'u aproximada (kPa)':u,'U (kN/m)':U,
                     'W sen α (kN/m)':W*sn,'Resistencia Fellenius (kN/m)':resisting})
    return fell,bishop,R,base,edges,df,it+1

st.title('⛰️ GEO-RETO 360 · Estabilidad de taludes')
st.caption('Simulador académico interactivo · Fellenius y Bishop simplificado · Caso 5 (Chosica)')
st.warning('Modelo didáctico de equilibrio límite para una sola superficie circular de prueba. No busca la superficie crítica ni reemplaza un estudio geotécnico. La presión de poros ru es una aproximación.')
with st.sidebar:
    st.header('Parámetros del talud')
    H=st.number_input('Altura H (m)',min_value=0.1,value=12.0,step=0.5)
    ratio=st.number_input('Pendiente horizontal/vertical (H:V)',min_value=0.1,value=1.5,step=0.1)
    gamma=st.number_input('Peso unitario γ (kN/m³)',min_value=0.1,value=18.0,step=0.5)
    c=st.number_input('Cohesión c (kPa)',min_value=0.0,value=15.0,step=1.0)
    phi=st.number_input('Ángulo de fricción φ (°)',min_value=0.0,max_value=59.9,value=24.0,step=1.0)
    n=st.slider('Número de dovelas',5,100,30,1)
    st.header('Superficie circular de prueba')
    xe=st.number_input('Salida horizontal x (m)',min_value=0.1,value=28.0,step=1.0)
    cx=st.number_input('Centro X (m)',value=8.0,step=1.0)
    auto=st.checkbox('Ajustar centro Y automáticamente',value=True)
    if auto:
        cy=(xe**2+H**2-2*xe*cx)/(2*H)
        st.write(f'Centro Y calculado: **{cy:.4f} m**')
    else:
        cy=st.number_input('Centro Y (m)',value=20.0,step=1.0)
    ru=st.slider('Coeficiente de presión de poros ru (aproximado)',0.0,0.9,0.0,0.05)

try:
    fell,bishop,R,base,edges,df,iterations=calcular(H,ratio,gamma,c,phi,n,xe,cx,cy,ru)
except ValueError as exc:
    st.error(str(exc));st.stop()

c1,c2,c3=st.columns(3)
c1.metric('FS Fellenius',f'{fell:.3f}')
c2.metric('FS Bishop simplificado',f'{bishop:.3f}')
c3.metric('Radio del círculo',f'{R:.2f} m')
fig,ax=plt.subplots(figsize=(11,5.5))
xx=np.linspace(-4,xe+6,600); xs=np.linspace(0,xe,600)
ax.plot(xx,terreno(xx,H,ratio),color='#70523b',lw=2.5,label='Perfil del terreno')
ax.plot(xs,base(xs),color='#c34238',lw=2.4,label='Superficie circular de prueba')
ax.fill_between(xs,base(xs),terreno(xs,H,ratio),color='#d8ae70',alpha=.45)
for x in edges[1:-1]:
    ax.plot([x,x],[float(base(x)),float(terreno(x,H,ratio))],color='#96754f',lw=.65)
ax.plot(cx,cy,'k+',ms=10,label='Centro de rotación')
ax.set_xlabel('Distancia horizontal (m)');ax.set_ylabel('Elevación (m)')
ax.set_title(f'{n} dovelas · ru = {ru:.2f} · Fellenius = {fell:.3f} · Bishop = {bishop:.3f}')
ax.grid(alpha=.2);ax.legend(loc='best',fontsize=8);ax.set_aspect('equal',adjustable='datalim')
fig.tight_layout();st.pyplot(fig)

with st.expander('Tabla detallada de dovelas',expanded=True):
    st.dataframe(df.round(4),use_container_width=True,hide_index=True)

params={'Altura H (m)':H,'Pendiente H:V':ratio,'Peso unitario (kN/m³)':gamma,'Cohesión (kPa)':c,
        'Fricción (°)':phi,'Dovelas':n,'Salida x (m)':xe,'Centro X (m)':cx,'Centro Y (m)':cy,
        'Radio (m)':R,'ru aproximado':ru,'FS Fellenius':fell,'FS Bishop simplificado':bishop,
        'Iteraciones Bishop':iterations}
output=BytesIO()
with pd.ExcelWriter(output,engine='openpyxl') as writer:
    pd.DataFrame(list(params.items()),columns=['Parámetro','Valor']).to_excel(writer,index=False,sheet_name='Resumen')
    df.to_excel(writer,index=False,sheet_name='Dovelas')
st.download_button('⬇️ Descargar resultados en Excel',data=output.getvalue(),file_name='GEO_RETO_360_resultados.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
st.info('El valor ru se aproxima como u = ru · γ · altura de suelo sobre la base. No corresponde a un análisis de filtración. Los datos de geometría circular son supuestos académicos editables porque la guía no proporciona centro y radio numéricos.')
