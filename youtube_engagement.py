import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; 
warnings.filterwarnings(action='ignore')

# warning 삭제용
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import plotly.express as px
import math

class YoutubeViz():
    def __init__(self,df):
        self.df = df
        self.df['published_date'] = pd.to_datetime(self.df['published_date'])
        self.df['year_month'] = self.df['published_date'].dt.strftime('%Y-%m')
        self.df['week_num'] = self.df['published_date'].dt.week
        # q1 해결을 위한 요약통계량
        self.q1 = self.df[['video_id','category_name','channel_id','year_month','week_num']].groupby(['category_name','channel_id']). \
            count().\
            sort_values(by='video_id',ascending=False).\
            reset_index(drop=False)
        # tag list column 생성
        self.df['tags_split'] = self.df['tags'].str.split('|')
        # 월별 주차 계산
        self.df['week_num_month'] = self.df['published_date'].apply(self._week_of_month)
        # n월 n주차 변수
        self.df['month_week'] = self.df['year_month'].astype(str) + '-'+ self.df['week_num_month'].astype(str)
        self.df['month_week'] = self.df['month_week'].apply(lambda x: x.split('-')[1]+"월 "+x.split('-')[2]+"주")
        
        
    def _week_of_month(self,dt):
        first_day = dt.replace(day=1)
        dom = dt.day
        adjusted_dom = dom + first_day.weekday()
    
        return int(math.ceil(adjusted_dom/7.0))
    
    # list of list 형태의 Series를 태그의 개수를 세기 위한 함수
    def _count_tag(self,tags:pd.Series):
        """
        series.explode().value_counts()
        """
        tags.dropna(inplace=True)
        res=pd.Series(sum([item for item in tags], [])).value_counts().sort_values(ascending=False)
        
        return res
    
    # 카테고리별 채널별 비디오 개수
    def solve_q1_1(self):
        fig = px.bar(self.q1, y='video_id',x='channel_id' ,color='category_name',
                     orientation='v',barmode='stack',
                     color_discrete_sequence=px.colors.qualitative.Dark24,
                     text='video_id')
        fig.update_layout(title_text='Video Count by Category and Channel',
                        xaxis_title='Channel',
                        yaxis_title='Video Count',
                        width=1600,
                        height=800)
        fig.write_html('q1_1.html')
            
        return fig
    
    # 월별 카테고리별 채널별 비디오 개수
    def solve_q1_2(self):
        self.q1_2 = self.df[['video_id','category_name','channel_id','year_month']].groupby(['year_month','category_name','channel_id']). \
            count().\
            sort_values(by='video_id', ascending=False).\
            reset_index(drop=False)
        fig = px.bar(self.q1_2, y='video_id',x='channel_id' ,color='category_name',facet_row="year_month",
                    orientation='v',barmode='stack',facet_col_wrap=2,
                    color_discrete_sequence=px.colors.qualitative.Light24,
                    text='video_id')

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.for_each_yaxis(lambda y: y.update(title='Video Count'))
        fig.update_layout(title_text='Video Count by Category and Channel by Month',
                        xaxis_title='Channel',
                        #yaxis_title='Video Count',
                        width=1600,
                        height=800)
        fig.write_html('q1_2.html')
        
        return fig

    # 월별 비디오 개수 기준 Top10 채널
    def solve_q1_3(self):
        self.q1_3 = self.df[['video_id','category_name','channel_id','year_month']].groupby(['year_month','channel_id','category_name']). \
            count().\
            sort_values(by='video_id', ascending=False).\
            reset_index(drop=False)
        top10 = self.q1_3.sort_values(['year_month','video_id'],ascending=[True,False]).groupby('year_month'). \
            head(10)
        fig = px.bar(top10, y='video_id',x='channel_id' ,facet_col="year_month",color='category_name',color_discrete_sequence=px.colors.qualitative.Dark24,text='video_id')
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.for_each_xaxis(lambda x: x.update(title=''))
        fig.update_xaxes(showticklabels=True , matches=None)
        fig.update_layout(title_text='Top 10 Channel by Month',
                        yaxis_title='Video Count',
                        xaxis_title=None)
        fig.write_html('q1_3.html')
    
        return fig
        
    # 주별 비디오 개수 기준 Top5 채널
    def solve_q1_4(self):
        self.q1_4 = self.df[['video_id','category_name','channel_id','week_num','month_week']].groupby(['month_week','week_num','channel_id','category_name']). \
            count().\
            sort_values(by='video_id',ascending=False).\
            reset_index(drop=False) 
        top5=self.q1_4.sort_values(['month_week','video_id'],ascending=[True,False]).groupby('month_week'). \
            head(5)
        fig = px.bar(top5, y='video_id',x='channel_id' ,facet_col="month_week",text='video_id',color='category_name',color_discrete_sequence=px.colors.qualitative.Dark24)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.for_each_xaxis(lambda x: x.update(title=''))
        fig.update_xaxes(showticklabels=True , matches=None)
        #fig.update_xaxes(showticklabels=True , matches=None)
        # hide the x-axis title
        fig.update_layout(title_text='Top 5 Channel by Week',
                        yaxis_title='Video Count',
                        xaxis_title=None)
        fig.write_html('q1_4.html')
        
    
        
        return fig
    

    # 월별 카테고리별 태그 키워드 순위 시각화
    def solve_q1_5(self):
        q1_5 = self.df.groupby(['year_month','category_name'])['tags_split'].apply(self._count_tag).reset_index()
        q1_5.sort_values(['year_month','category_name','tags_split'],ascending=[True,True,False],inplace=True)
        q1_5_viz=q1_5.groupby(['year_month','category_name']).head(5)
        
        fig_tag = px.bar(q1_5_viz, y='tags_split',x='level_2' ,facet_col="year_month", facet_row="category_name",
                labels={'year_month':'Year-Month','category_name':'Category','tags_split':'Tag Count','level_2':'Tag'},text='tags_split')
        fig_tag.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig_tag.for_each_annotation(lambda a: a.update(borderpad=29,borderwidth=15, font= dict(size=15,color='black')))
        # annotation category position
        fig_tag.update_xaxes(showticklabels=True , matches=None)
        fig_tag.update_layout(title_text='',
                    # yaxis margin
                    yaxis = dict(automargin=True),
                    # annotaion margin
                    margin=dict(l=80,r=80,b=80,t=80),
                    width=1600,
                    height=2400)
        #fig_tag.update_layout(title_pad=dict(t=5,l=10,b=50,r=10))
        fig_tag.write_image('q1_5.png')
        
        
        return fig_tag


from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import numpy as np

# clsss YouTubeEngagement 계산을 위한 클래스
class YoutubeEngagement():
    
    def __init__(self, df):
        self.df = df
        self.df['playtime'] = self.df['duration'].apply(lambda x: x.replace('PT',''))
        self.df['published_date'] = pd.to_datetime(self.df['published_date'])
        self.df['on_trending_date'] = pd.to_datetime(self.df['on_trending_date'])
        self.df['off_trending_date'] = pd.to_datetime(self.df['off_trending_date'])
    
    # %h%m%s format을 시간으로 변환하기 위한 함수    
    def _time_parser(self,s:str):
        s = s.lower()
        fmt=''.join('%'+c.upper()+c for c in 'hms' if c in s)
        return datetime.strptime(s, fmt).time()
    # 시간을 초로 변환하는 함수
    def _time_to_sec(self,t:datetime):
        return t.hour*3600 + t.minute*60 + t.second
    
    def engagement(self):
        self.df['playtime_sec'] = self.df['playtime'].apply(self._time_parser).apply(self._time_to_sec)
        #self.df['publish_trend_datediff'] = (self.df['on_trending_date'] - self.df['published_date']).apply(lambda x: x.days) # smaller the better
        self.df['trend_off_datediff'] = (self.df['off_trending_date'] - self.df['on_trending_date']).apply(lambda x: x.days)
        self.df['rank_diff'] = self.df['off_rank'] - self.df['on_rank'] # bigger the better
        # view, like, comment, subscribe 
        self.df['views_diff'] = (self.df['off_views'] - self.df['on_views']) #* 100 /self.df['on_views'] # 
        self.df['likes_diff'] = (self.df['off_likes'] - self.df['on_likes']) #* 100 /self.df['on_likes'] #
        # dislike를 포함시킬 것인가?
        #self.df['dislike_diff'] = (self.df['off_dislikes'] - self.df['on_dislikes']) * 100 /self.df['on_dislikes']        
        self.df['comments_diff'] = (self.df['off_comments'] - self.df['on_comments']) #* 100 /self.df['on_comments'] #
        self.df['subscribers_diff'] = (self.df['off_channel_subscribers'] - self.df['on_channel_subscribers']) #* 100 /self.df['on_channel_subscribers'] #        
        self.df['engagement_score'] = self.df['trend_off_datediff'] + self.df['rank_diff'] + self.df['views_diff'] + self.df['likes_diff'] + self.df['comments_diff'] + self.df['subscribers_diff'] - self.df['publish_trend_datediff']
    
        return self.df
            

class Vizcorr():
  def __init__(self):
    """
    engagement score와 관련변수의 상관관계 시각화를 위한 class
    """
    pass
    
  def corrdot(self, *args, **kwargs):
    """
    상관계수 text annotation
    """
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = round(corr_r, 3)
    ax = plt.gca()
    font_size = abs(corr_r) * 80 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)
    
  def corrfunc(self, x, y, **kwargs):
    """
    p value text annotation 용 함수
    """
    r, p = stats.pearsonr(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate(p_stars, xy=(0.65, 0.6), xycoords=ax.transAxes,
                color='red', fontsize=55)
    
  def corr_plot(self,df:pd.DataFrame):
    sns.set(style="white")
    g = sns.PairGrid(df, aspect=1.5, diag_sharey=False, despine=False)
    g.map_lower(sns.regplot, lowess=True, ci=False,
                line_kws={'color': 'red', 'lw': 1},
                scatter_kws={'color': 'black', 's': 20})
    g.map_diag(sns.histplot, color='black',kde=True) 
    g.map_diag(sns.rugplot, color='black')
    g.map_upper(self.corrdot)
    g.map_upper(self.corrfunc)
    g.fig.subplots_adjust(wspace=0, hspace=0)
    
    # axis labels 삭제
    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')

    
    for ax, col in zip(np.diag(g.axes), df.columns):
      ax.set_title(col, y=0.82, fontsize=26)
      
      
    return g
  
  
  def get_redundant_pairs(self,df):
    """
    상관행렬의 하단 상관계수 패어만 추출
    """
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop
  

  def get_top_abs_corr(self,df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = self.get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
  

