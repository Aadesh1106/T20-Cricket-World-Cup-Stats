import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🏏 T20 World Cup Match Analysis",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Team logos dictionary - using more reliable sources
TEAM_LOGOS = {
    "India": "https://flagcdn.com/w40/in.png",
    "England": "https://flagcdn.com/w40/gb-eng.png",
    "Pakistan": "https://flagcdn.com/w40/pk.png",
    "New Zealand": "https://flagcdn.com/w40/nz.png",
    "South Africa": "https://flagcdn.com/w40/za.png",
    "Bangladesh": "https://flagcdn.com/w40/bd.png",
    "Sri Lanka": "https://flagcdn.com/w40/lk.png",
    "West Indies": "https://flagcdn.com/w40/bb.png",  # Using Barbados as representative
    "Afghanistan": "https://flagcdn.com/w40/af.png",
    "Ireland": "https://flagcdn.com/w40/ie.png",
    "Scotland": "https://flagcdn.com/w40/gb-sct.png",
    "Netherlands": "https://flagcdn.com/w40/nl.png",
    "UAE": "https://flagcdn.com/w40/ae.png",
    "Zimbabwe": "https://flagcdn.com/w40/zw.png",
    "Kenya": "https://flagcdn.com/w40/ke.png",
    "Canada": "https://flagcdn.com/w40/ca.png",
}

class T20WorldCupAnalyzer:
    def __init__(self):
        self.df = None
        self.filtered_df = None
        
    @st.cache_data
    def load_data(_self, file_path="t20_worldcup_matches.csv"):
        """Load and preprocess the data with error handling"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            
            # Data preprocessing
            df = _self._preprocess_data(df)
            return df
            
        except FileNotFoundError:
            st.error(f"❌ File '{file_path}' not found. Please ensure the CSV file is in the correct location.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.stop()
    
    def _preprocess_data(self, df):
        """Preprocess the dataframe"""
        # Handle date column
        date_columns = ['date', 'match_date', 'Date']
        date_col = None
        
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
        else:
            df['year'] = None
            df['month'] = None
        
        # Clean team names
        team_columns = ['team1', 'team2', 'winner']
        for col in team_columns:
            if col in df.columns:
                df[col] = df[col].str.strip()
        
        # Handle missing values
        df = df.fillna({'player_of_match': 'Unknown', 'venue': 'Unknown Venue'})
        
        # Calculate run differences if run columns exist
        if all(col in df.columns for col in ['team1_runs', 'team2_runs']):
            df['run_difference'] = abs(df['team1_runs'] - df['team2_runs'])
            df['total_runs'] = df['team1_runs'] + df['team2_runs']
            df['max_runs'] = df[['team1_runs', 'team2_runs']].max(axis=1)
        
        return df
    
    def get_team_stats(self, team_name):
        """Get comprehensive statistics for a team"""
        team_matches = self.df[
            (self.df['team1'] == team_name) | (self.df['team2'] == team_name)
        ]
        
        wins = len(team_matches[team_matches['winner'] == team_name])
        total_matches = len(team_matches)
        win_rate = (wins / total_matches * 100) if total_matches > 0 else 0
        
        return {
            'total_matches': total_matches,
            'wins': wins,
            'losses': total_matches - wins,
            'win_rate': win_rate
        }
    
    def create_enhanced_bar_chart(self, data, x_col, y_col, title, color_scale='Viridis'):
        """Create enhanced bar chart with better styling"""
        fig = px.bar(
            data,
            x=x_col,
            y=y_col,
            title=title,
            color=y_col,
            color_continuous_scale=color_scale,
            text=y_col
        )
        
        # Safe string formatting for hover template
        y_label = str(y_col).replace('_', ' ').title() if '_' in str(y_col) else str(y_col).title()
        x_label = str(x_col).replace('_', ' ').title() if '_' in str(x_col) else str(x_col).title()
        
        fig.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            hovertemplate=f"<b>%{{x}}</b><br>{y_label}: %{{y}}<extra></extra>"
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title=x_label,
            yaxis_title=y_label,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def display_team_with_flag(self, team_name):
        """Display team name with flag"""
        flag_url = TEAM_LOGOS.get(team_name, "")
        if flag_url:
            return f'<img src="{flag_url}" width="20" style="vertical-align:middle;margin-right:8px;">{team_name}'
        return team_name
    
    def create_matches_table(self, df_subset, max_rows=100):
        """Create an enhanced matches table with pagination"""
        if len(df_subset) > max_rows:
            st.warning(f"Showing first {max_rows} matches out of {len(df_subset)} total matches.")
            df_subset = df_subset.head(max_rows)
        
        # Create display dataframe with flags
        display_df = df_subset.copy()
        
        for col in ['team1', 'team2', 'winner']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(self.display_team_with_flag)
        
        # Format date if available
        if 'date' in display_df.columns:
            display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
        
        return display_df
    
    def generate_insights(self):
        """Generate key insights from the data"""
        insights = []
        
        # Most successful team
        winner_counts = self.filtered_df['winner'].value_counts()
        if not winner_counts.empty:
            most_successful = winner_counts.index[0]
            insights.append(f"🏆 **Most Successful Team**: {most_successful} with {winner_counts.iloc[0]} wins")
        
        # Best player
        player_counts = self.filtered_df['player_of_match'].value_counts()
        if not player_counts.empty and player_counts.index[0] != 'Unknown':
            best_player = player_counts.index[0]
            insights.append(f"⭐ **Top Performer**: {best_player} with {player_counts.iloc[0]} Player of the Match awards")
        
        # Venue analysis
        venue_counts = self.filtered_df['venue'].value_counts()
        if not venue_counts.empty and venue_counts.index[0] != 'Unknown Venue':
            popular_venue = venue_counts.index[0]
            insights.append(f"🏟️ **Most Popular Venue**: {popular_venue} ({venue_counts.iloc[0]} matches)")
        
        return insights

# Initialize the analyzer
@st.cache_resource
def get_analyzer():
    return T20WorldCupAnalyzer()

analyzer = get_analyzer()

# Load data
try:
    with st.spinner("Loading T20 World Cup data..."):
        analyzer.df = analyzer.load_data()
except Exception as e:
    st.error(f"Failed to initialize application: {str(e)}")
    st.stop()

# Get unique values for filters
unique_teams = pd.unique(analyzer.df[['team1', 'team2']].values.ravel('K'))
team_list = sorted([team for team in unique_teams if pd.notna(team)])

years = sorted(analyzer.df['year'].dropna().unique().astype(int)) if 'year' in analyzer.df.columns else []
venues = sorted(analyzer.df['venue'].dropna().unique()) if 'venue' in analyzer.df.columns else []

# Header
st.markdown('<h1 class="main-header">🏏 T20 World Cup Match Analysis</h1>', unsafe_allow_html=True)
st.markdown("### Comprehensive analysis of T20 World Cup matches with interactive visualizations")

# Sidebar
st.sidebar.header("🔧 Filters & Controls")
st.sidebar.markdown("---")

# Filters
selected_year = st.sidebar.selectbox(
    "📅 Select Year", 
    ["All"] + [str(y) for y in years],
    help="Filter matches by year"
)

selected_venue = st.sidebar.selectbox(
    "🏟️ Select Venue", 
    ["All"] + venues,
    help="Filter matches by venue"
)

selected_team = st.sidebar.selectbox(
    "🏏 Select Team", 
    ["All"] + team_list,
    help="Filter matches involving specific team"
)

# Apply filters
def apply_filters(df):
    temp_df = df.copy()
    if selected_year != "All":
        temp_df = temp_df[temp_df['year'] == int(selected_year)]
    if selected_venue != "All":
        temp_df = temp_df[temp_df['venue'] == selected_venue]
    if selected_team != "All":
        temp_df = temp_df[(temp_df['team1'] == selected_team) | (temp_df['team2'] == selected_team)]
    return temp_df

analyzer.filtered_df = apply_filters(analyzer.df)

# Key metrics
st.markdown("## 📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏟️ Total Matches", len(analyzer.filtered_df))

with col2:
    unique_teams_filtered = len(pd.unique(analyzer.filtered_df[['team1', 'team2']].values.ravel('K')))
    st.metric("🌍 Teams", unique_teams_filtered)

with col3:
    unique_players = analyzer.filtered_df['player_of_match'].nunique()
    st.metric("⭐ Unique POTM", unique_players)

with col4:
    unique_venues = analyzer.filtered_df['venue'].nunique()
    st.metric("🏟️ Venues", unique_venues)

st.markdown("---")

# Key Insights
st.markdown("## 💡 Key Insights")
insights = analyzer.generate_insights()
for insight in insights:
    st.markdown(insight)

st.markdown("---")

# Wins Analysis
st.markdown("## 🏆 Team Performance Analysis")

if not analyzer.filtered_df.empty:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Wins chart
        win_counts = analyzer.filtered_df['winner'].value_counts().reset_index()
        win_counts.columns = ['team', 'wins']
        
        fig = analyzer.create_enhanced_bar_chart(
            win_counts, 'team', 'wins', 
            'Number of Wins by Team', 
            'Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Team statistics
        st.markdown("### 📈 Team Stats")
        for _, row in win_counts.head(5).iterrows():
            team_stats = analyzer.get_team_stats(row['team'])
            st.markdown(f"""
            **{row['team']}**
            - Wins: {team_stats['wins']}
            - Win Rate: {team_stats['win_rate']:.1f}%
            """)

else:
    st.warning("No data available for the selected filters.")

st.markdown("---")

# Player Performance
st.markdown("## 🌟 Player Performance")

if 'player_of_match' in analyzer.filtered_df.columns:
    player_stats = analyzer.filtered_df['player_of_match'].value_counts().head(10).reset_index()
    player_stats.columns = ['player', 'awards']
    
    fig = analyzer.create_enhanced_bar_chart(
        player_stats, 'player', 'awards',
        'Top 10 Players - Most Player of the Match Awards',
        'Plasma'
    )
    
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Match Results Distribution
st.markdown("## 📊 Match Results Distribution")

if 'result' in analyzer.filtered_df.columns:
    result_counts = analyzer.filtered_df['result'].value_counts().reset_index()
    result_counts.columns = ['result_type', 'count']
    
    if not result_counts.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            pie_chart = px.pie(
                result_counts,
                names='result_type',
                values='count',
                title='Distribution of Match Results',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            pie_chart.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(pie_chart, use_container_width=True)
        
        with col2:
            bar_chart = px.bar(
                result_counts,
                x='count',
                y='result_type',
                orientation='h',
                title='Match Results - Count',
                color='count',
                color_continuous_scale='Blues'
            )
            bar_chart.update_layout(showlegend=False)
            st.plotly_chart(bar_chart, use_container_width=True)

st.markdown("---")

# Head-to-Head Comparison
st.markdown("## 🤝 Head-to-Head Analysis")

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Select Team 1", team_list, key='h2h_team1')
with col2:
    available_teams = [t for t in team_list if t != team1]
    team2 = st.selectbox("Select Team 2", available_teams, key='h2h_team2')

if team1 and team2:
    h2h_matches = analyzer.df[
        ((analyzer.df['team1'] == team1) & (analyzer.df['team2'] == team2)) |
        ((analyzer.df['team1'] == team2) & (analyzer.df['team2'] == team1))
    ]
    
    if not h2h_matches.empty:
        col1, col2, col3 = st.columns(3)
        
        team1_wins = len(h2h_matches[h2h_matches['winner'] == team1])
        team2_wins = len(h2h_matches[h2h_matches['winner'] == team2])
        total_matches = len(h2h_matches)
        
        with col1:
            st.metric(f"{team1} Wins", team1_wins)
        with col2:
            st.metric(f"{team2} Wins", team2_wins)
        with col3:
            st.metric("Total Matches", total_matches)
        
        # H2H visualization
        h2h_data = pd.DataFrame({
            'team': [team1, team2],
            'wins': [team1_wins, team2_wins]
        })
        
        fig_h2h = px.bar(
            h2h_data,
            x='team',
            y='wins',
            title=f"{team1} vs {team2} - Head to Head",
            color='wins',
            color_continuous_scale='RdYlBu'
        )
        fig_h2h.update_layout(showlegend=False)
        st.plotly_chart(fig_h2h, use_container_width=True)
        
        # Recent matches
        st.markdown("### Recent Matches")
        recent_matches = analyzer.create_matches_table(h2h_matches.tail(10))
        st.write(recent_matches.to_html(escape=False, index=False), unsafe_allow_html=True)
        
    else:
        st.info(f"No matches found between {team1} and {team2}")

st.markdown("---")

# Detailed Match Data
st.markdown("## 🔍 Detailed Match Data")

# Show filtered matches
if not analyzer.filtered_df.empty:
    display_df = analyzer.create_matches_table(analyzer.filtered_df)
    
    # Search functionality
    search_term = st.text_input("🔎 Search matches (team, venue, player):", "")
    
    if search_term:
        mask = display_df.astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]
    
    st.write(f"Showing {len(display_df)} matches")
    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    # Download button
    csv = analyzer.filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"t20_worldcup_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

else:
    st.info("No matches found for the selected filters.")

# Footer
st.markdown("---")
st.markdown("### 🙏 Thank you for using the T20 World Cup Analysis Dashboard!")
st.markdown("""
**Features:**
- 🔍 Advanced filtering and search
- 📊 Interactive visualizations
- 📈 Comprehensive team statistics
- 🤝 Head-to-head comparisons
- 📥 Data export capabilities

Built with ❤️ using **Streamlit** and **Plotly**
""")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 App Info")
st.sidebar.info(f"""
**Data Summary:**
- Total Matches: {len(analyzer.df)}
- Date Range: {analyzer.df['year'].min():.0f} - {analyzer.df['year'].max():.0f}
- Teams: {len(team_list)}
- Venues: {len(venues)}
""")

st.sidebar.markdown("### 🔄 Refresh Data")
if st.sidebar.button("Refresh"):
    st.cache_data.clear()

    st.rerun()
