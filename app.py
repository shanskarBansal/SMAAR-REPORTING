
import streamlit as st
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import time
from googleapiclient.discovery import build
from google.oauth2 import service_account
import tempfile
import json


creds_json = st.secrets["gcp_service_account"]
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
client = gspread.authorize(creds)


creds_jsons = dict(st.secrets["script_service_account"])
scoped = [
    'https://www.googleapis.com/auth/script.projects',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/documents',  
    'https://www.googleapis.com/auth/spreadsheets' 
    ]

with tempfile.NamedTemporaryFile() as temp_file:
    temp_file.write(json.dumps(creds_jsons).encode())
    temp_file.flush() 

    creddd = service_account.Credentials.from_service_account_file(
        temp_file.name, scopes=scoped)


def scripting(creds):
    delegated_creds = creds.with_subject('krishan.maggo@varaheanalytics.com') 
    service = build('script', 'v1', credentials=delegated_creds)
    script_id = '1gzDFr1oJTtAJeTv1uZIvLQe82IkIzWjh0_LT7IaOpPDUuLGKaFHYWvTH'
    function_name = 'processData'
    request = {
        'function': function_name,
        'devMode': True  
    }
    try:
        response = service.scripts().run(body=request, scriptId=script_id).execute()
        st.write('Function executed successfully:', response) 
    except Exception as e:
        st.write(f"Error executing Apps Script: {str(e)}")


def create_weekly_sheet_copy(client, spreadsheet, base_sheet_name):
    current_week_number = datetime.now().isocalendar().week
    new_sheet_name = f"Week_{current_week_number-1}"

    # Attempt to access the worksheet by name
    try:
        worksheet = spreadsheet.worksheet(new_sheet_name)
        return f"Worksheet {new_sheet_name} already exists.", False
    except gspread.WorksheetNotFound:
        # Create a new worksheet by duplicating the base if not found
        base_worksheet = spreadsheet.worksheet(base_sheet_name)
        new_worksheet = spreadsheet.duplicate_sheet(
            source_sheet_id=base_worksheet.id,
            new_sheet_name=new_sheet_name
        )
        return f"Created new worksheet: {new_sheet_name}",True


def remove_duplicate_posts(df,post_id):
    unique_df = df.drop_duplicates(subset= post_id, keep='first')
    return unique_df


def read_files(client,url,sheet_name):
    try:
        smaar_repo = client.open_by_url(url) 
        df = pd.DataFrame.from_dict(smaar_repo.worksheet(sheet_name).get_all_values())
        df = remove_first_row(df)
        return df
    except Exception as ex:
        print(ex)


def remove_first_row(df):
    new_header = df.iloc[0]  
    df = df[1:]  
    df.columns = new_header  
    df.reset_index(drop=True, inplace=True)
    return df


def post_stats(df,post_id,permalink,likes,comments,shares,reach,engagement,impressions):
    if len(df)==0:
        list1 = ['']*7
    else:
        df = remove_duplicate_posts(df,post_id)
        df[likes] = pd.to_numeric(df[likes], errors='coerce').fillna(0).astype(int)
        df[comments] = pd.to_numeric(df[comments], errors='coerce').fillna(0).astype(int)
        df[shares] = pd.to_numeric(df[shares], errors='coerce').fillna(0).astype(int)
        df[reach] = pd.to_numeric(df[reach], errors='coerce').fillna(0).astype(int)
        df[engagement] = pd.to_numeric(df[engagement], errors='coerce').fillna(0).astype(int)
        df[impressions] = pd.to_numeric(df[impressions], errors='coerce').fillna(0).astype(int)
        total_post = df[permalink].nunique()
        total_post_likes = df[likes].astype(int).sum()
        total_comments  = df[comments].astype(int).sum()
        total_shares  = df[shares].astype(int).sum()
        total_reach  = df[reach].astype(int).sum()
        total_engagment  = df[engagement].astype(int).sum()
        total_impression  = df[impressions].astype(int).sum()
        list1 = [total_post,total_post_likes,total_comments,total_shares,total_reach,total_engagment,total_impression]
    return list1


def content_performance(df,post_id,permalink,likes,comments,shares,reach,engagement,impressions):
    if len(df)==0:
        content = ['']*13
    else:
        df = remove_duplicate_posts(df,post_id)
        df[likes] = pd.to_numeric(df[likes], errors='coerce').fillna(0).astype(int)
        df[comments] = pd.to_numeric(df[comments], errors='coerce').fillna(0).astype(int)
        df[shares] = pd.to_numeric(df[shares], errors='coerce').fillna(0).astype(int)
        df[reach] = pd.to_numeric(df[reach], errors='coerce').fillna(0).astype(int)
        df[engagement] = pd.to_numeric(df[engagement], errors='coerce').fillna(0).astype(int)
        df[impressions] = pd.to_numeric(df[impressions], errors='coerce').fillna(0).astype(int)
        total_post = df[permalink].nunique()
        total_post_likes = df[likes].astype(int).sum()
        total_comments  = df[comments].astype(int).sum()
        total_shares  = df[shares].astype(int).sum()
        total_reach  = df[reach].astype(int).sum()
        total_engagement  = df[engagement].astype(int).sum()
        total_impressions  = df[impressions].astype(int).sum()
        avg_post_likes = int(df[likes].astype(int).mean())
        avg_comments  = int(df[comments].astype(int).mean())
        avg_shares  = int(df[shares].astype(int).mean())
        avg_reach  = int(df[reach].astype(int).mean())
        avg_engagement  = int(df[engagement].astype(int).mean())
        avg_impression  = int(df[impressions].astype(int).mean())
        content = [total_post,total_post_likes,avg_post_likes,total_comments,avg_comments,total_shares,avg_shares,
                    total_reach,avg_reach,total_engagement,avg_engagement,total_impressions,avg_impression]
    return content


def top_bottom_posts(df,post_id,reach,post_type,permalink,likes,comments,shares,engagements,impressions,message):
    if len(df)==0:
        flat_top_bottom_content = ['']*54
    else:
        df = remove_duplicate_posts(df,post_id)
        max_page_reach_idx = pd.to_numeric(df[reach], errors='coerce').fillna(0).astype(int).idxmax()
        second_largest_index = pd.to_numeric(df[reach], errors='coerce').fillna(0).astype(int).nlargest(2).index[-1]
        third_largest_index = pd.to_numeric(df[reach], errors='coerce').fillna(0).astype(int).nlargest(3).index[-1]
        rank1_post_link = df.loc[max_page_reach_idx, [post_type,permalink,message, likes,comments, shares,reach,engagements,impressions]]
        rank2_post_link = df.loc[second_largest_index,[ post_type,permalink,message, likes,comments, shares,reach,engagements,impressions]]
        rank3_post_link = df.loc[third_largest_index, [post_type,permalink,message, likes,comments, shares,reach,engagements,impressions]]
        # FB last three content data
        last_page_reach_idx = pd.to_numeric(df[reach], errors='coerce').fillna(0).astype(int).idxmin()
        second_smallest_index = pd.to_numeric(df[reach], errors='coerce').fillna(0).astype(int).nsmallest(2).index[-1]
        third_smallest_index = pd.to_numeric(df[reach], errors='coerce').fillna(0).astype(int).nsmallest(3).index[-1]
        last1_post_link = df.loc[last_page_reach_idx, [post_type,permalink,message, likes,comments, shares,reach,engagements,impressions]]
        last2_post_link = df.loc[second_smallest_index, [post_type,permalink,message, likes,comments, shares,reach,engagements,impressions]]
        last3_post_link = df.loc[third_smallest_index, [post_type,permalink,message, likes,comments, shares,reach,engagements,impressions]]
        flat_top_bottom_content = rank1_post_link.values.tolist() + rank2_post_link.values.tolist() + rank3_post_link.values.tolist() + last3_post_link.values.tolist() + last2_post_link.values.tolist() + last1_post_link.values.tolist()
    return flat_top_bottom_content

def calculate_growth_with_conditions(df, numerator_col, denominator_col, result_col):
    df[numerator_col] = pd.to_numeric(df[numerator_col].replace(' ', '0'), errors='coerce').fillna(0)
    df[denominator_col] = pd.to_numeric(df[denominator_col].replace(' ', '0'), errors='coerce').fillna(0)
    
    # Initialize the result column with '0%'
    df[result_col] = '0%'
    
    # Mask for rows where denominator is not 0 (to avoid division by zero)
    valid_mask = df[denominator_col] != 0
    
    # Perform the calculation where valid
    df.loc[valid_mask, result_col] = (((df.loc[valid_mask, numerator_col] - df.loc[valid_mask, denominator_col]) /
                                        df.loc[valid_mask, denominator_col]) * 100).round().astype(int).astype(str) + '%'

# assign the columns name

headers_of_fb_top_bottom_content = ['rank1_fb_post_type','rank1_fb_post_link','rank1_fb_post_caption','rank1_fb_post_likes','rank1_fb_post_comments','rank1_fb_post_shares','rank1_fb_post_reach','rank1_fb_post_engagement','rank1_fb_post_impression',
                                    'rank2_fb_post_type','rank2_fb_post_link','rank2_fb_post_caption','rank2_fb_post_likes','rank2_fb_post_comments','rank2_fb_post_shares','rank2_fb_post_reach','rank2_fb_post_engagement','rank2_fb_post_impression',
                                    'rank3_fb_post_type','rank3_fb_post_link','rank3_fb_post_caption','rank3_fb_post_likes','rank3_fb_post_comments','rank3_fb_post_shares','rank3_fb_post_reach','rank3_fb_post_engagement','rank3_fb_post_impression',
                                    'last3_fb_post_type','last3_fb_post_link','last3_fb_post_caption','last3_fb_post_likes','last3_fb_post_comments','last3_fb_post_shares','last3_fb_post_reach','last3_fb_post_engagement','last3_fb_post_impression',
                                    'last2_fb_post_type','last2_fb_post_link','last2_fb_post_caption','last2_fb_post_likes','last2_fb_post_comments','last2_fb_post_shares','last2_fb_post_reach','last2_fb_post_engagement','last2_fb_post_impression',
                                    'last1_fb_post_type','last1_fb_post_link','last1_fb_post_caption','last1_fb_post_likes','last1_fb_post_comments','last1_fb_post_shares','last1_fb_post_reach','last1_fb_post_engagement','last1_fb_post_impression']
headers_of_insta_top_bottom_content = ['rank1_insta_post_type','rank1_insta_post_link','rank1_insta_post_caption','rank1_insta_post_likes','rank1_insta_post_comments','rank1_insta_post_shares','rank1_insta_post_reach','rank1_insta_post_engagement','rank1_insta_post_impression',
                                    'rank2_insta_post_type','rank2_insta_post_link','rank2_insta_post_caption','rank2_insta_post_likes','rank2_insta_post_comments','rank2_insta_post_shares','rank2_insta_post_reach','rank2_insta_post_engagement','rank2_insta_post_impression',
                                    'rank3_insta_post_type','rank3_insta_post_link','rank3_insta_post_caption','rank3_insta_post_likes','rank3_insta_post_comments','rank3_insta_post_shares','rank3_insta_post_reach','rank3_insta_post_engagement','rank3_insta_post_impression',
                                    'last3_insta_post_type','last3_insta_post_link','last3_insta_post_caption','last3_insta_post_likes','last3_insta_post_comments','last3_insta_post_shares','last3_insta_post_reach','last3_insta_post_engagement','last3_insta_post_impression',
                                    'last2_insta_post_type','last2_insta_post_link','last2_insta_post_caption','last2_insta_post_likes','last2_insta_post_comments','last2_insta_post_shares','last2_insta_post_reach','last2_insta_post_engagement','last2_insta_post_impression',
                                    'last1_insta_post_type','last1_insta_post_link','last1_insta_post_caption','last1_insta_post_likes','last1_insta_post_comments','last1_insta_post_shares','last1_insta_post_reach','last1_insta_post_engagement','last1_insta_post_impression']

headers_of_fb_stats = ['current_week_fb_total_post','current_week_fb_total_likes','current_week_fb_total_comments','current_week_fb_total_shares','current_week_fb_total_reach','current_week_fb_total_engagement','current_week_fb_total_impression',
                        'previous_week_fb_total_post','previous_week_fb_total_likes','previous_week_fb_total_comments','previous_week_fb_total_shares','previous_week_fb_total_reach','previous_week_fb_total_engagement','previous_week_fb_total_impression']

headers_of_insta_stats = ['current_week_insta_total_post','current_week_insta_total_likes','current_week_insta_total_comments','current_week_insta_total_shares','current_week_insta_total_reach','current_week_insta_total_engagement','current_week_insta_total_impression',
                            'previous_week_insta_total_post','previous_week_insta_total_likes','previous_week_insta_total_comments','previous_week_insta_total_shares','previous_week_insta_total_reach','previous_week_insta_total_engagement','previous_week_insta_total_impression']


headers_of_fb_content_owner_performance = ['fb_varahe_number_of_post','fb_varahe_total_likes','fb_varahe_avg_likes','fb_varahe_total_comments','fb_varahe_avg_comments','fb_varahe_total_share','fb_varahe_avg_share','fb_varahe_total_reach','fb_varahe_avg_reach','fb_varahe_total_engagement','fb_varahe_avg_engagement','fb_varahe_total_impression','fb_varahe_avg_impression',
                                            'fb_party_number_of_post','fb_party_total_likes','fb_party_avg_likes','fb_party_total_comments','fb_party_avg_comments','fb_party_total_share','fb_party_avg_share','fb_party_total_reach','fb_party_avg_reach','fb_party_total_engagement','fb_party_avg_engagement','fb_party_total_impression','fb_party_avg_impression']

headers_of_insta_content_owner_performance = ['insta_varahe_number_of_post','insta_varahe_total_likes','insta_varahe_avg_likes','insta_varahe_total_comments','insta_varahe_avg_comments','insta_varahe_total_share','insta_varahe_avg_share','insta_varahe_total_reach','insta_varahe_avg_reach','insta_varahe_total_engagement','insta_varahe_avg_engagement','insta_varahe_total_impression','insta_varahe_avg_impression',
                                                'insta_party_number_of_post','insta_party_total_likes','insta_party_avg_likes','insta_party_total_comments','insta_party_avg_comments','insta_party_total_share','insta_party_avg_share','insta_party_total_reach','insta_party_avg_reach','insta_party_total_engagement','insta_party_avg_engagement','insta_party_total_impression','insta_party_avg_impression']

# Statis Data Created
today = pd.to_datetime('today')
last_week_date = today - timedelta(days=7)
prior_to_last_week_date = last_week_date - timedelta(days=7)
last_week_number = last_week_date.isocalendar().week
prior_to_last_week_number = prior_to_last_week_date.isocalendar().week
current_year = datetime.now().year
date_format = "%d %b %Y"
current_date = datetime.now()
day_of_week = current_date.weekday()
days_to_last_monday = day_of_week + 7
last_monday_date = (current_date - timedelta(days=days_to_last_monday)).strftime(date_format)
last_sunday_date = (current_date - timedelta(days=days_to_last_monday - 6)).strftime(date_format)


def main():
    st.title('Official Report Generator')
    repo_sheet_path = st.text_input("Give the current week tracker sheet link here")
    repo_sheet_name = st.text_input("Give the tracker sheet name here")
    st.write('Upload the current week FB and Insta post Data')
    current_week_link =st.text_input('Please enter the current week offical data worksheet link here')
    current_week_fb_sheet_name = st.text_input('Please enter the current week Facebook offical data worksheet tab Name here ')
    current_week_IG_sheet_name = st.text_input('Please enter the current week Instagram offical data worksheet tab Name here ')
    st.write('Upload the Previous week FB and Insta post Data')
    previous_week_link =st.text_input('Please enter the previous week offical data worksheet link here')
    previous_week_fb_sheet_name = st.text_input('Please enter the previous week Facebook offical data worksheet tab Name here ')
    previous_week_IG_sheet_name = st.text_input('Please enter the previous week Instagram offical data worksheet tab Name here ')
    st.write("Give the information on Google workbook where you want horizontal to be stored")
    posting_sheet_link = st.text_input("Posting sheet link")
    posting_sheet_name = st.text_input("Posting sheet Name")
    if repo_sheet_path and repo_sheet_name and current_week_link and current_week_fb_sheet_name and current_week_IG_sheet_name and previous_week_link and previous_week_fb_sheet_name and previous_week_IG_sheet_name and posting_sheet_link and posting_sheet_name:
        process_button = st.button('Process Sheet')
        if process_button:
            run_code(repo_sheet_path, repo_sheet_name, current_week_link, current_week_fb_sheet_name, current_week_IG_sheet_name, previous_week_link, previous_week_fb_sheet_name, previous_week_IG_sheet_name,posting_sheet_link,posting_sheet_name)

def run_code(repo_sheet_path, repo_sheet_name, current_week_link, current_week_fb_sheet_name, current_week_IG_sheet_name, previous_week_link, previous_week_fb_sheet_name, previous_week_IG_sheet_name,posting_sheet_link,posting_sheet_name):
    official_fb_data = read_files(client,current_week_link,current_week_fb_sheet_name)
    official_repo_data = read_files(client,repo_sheet_path,repo_sheet_name)
    official_insta_data = read_files(client,current_week_link,current_week_IG_sheet_name)
    official_previous_fb_data = read_files(client,previous_week_link,previous_week_fb_sheet_name)
    official_previous_insta_data = read_files(client,previous_week_link,previous_week_IG_sheet_name)
    result_df = pd.DataFrame()
    fb_stats_df = pd.DataFrame(columns=headers_of_fb_stats)
    insta_stats_df = pd.DataFrame(columns=headers_of_insta_stats)
    fb_top_bottom_df = pd.DataFrame(columns=headers_of_fb_top_bottom_content)
    insta_top_bottom_df = pd.DataFrame(columns=headers_of_insta_top_bottom_content)
    insta_content_performance_df = pd.DataFrame(columns=headers_of_insta_content_owner_performance)
    fb_content_performance_df = pd.DataFrame(columns=headers_of_fb_content_owner_performance)
    initial_df = pd.DataFrame(columns=['State_name','Page_name','Week Range'])
    for i,row in official_repo_data.iterrows():
        state_name = row['State']
        property_name = row['Official Party Page name']
        fb_post_filter = row['FB Page Name']
        insta_post_filter = row['Insta Account Name']
        Week_range_name = last_monday_date + ' to ' + last_sunday_date
        initial_list = [state_name,property_name,Week_range_name]
        #this week filter 
        filter_current_week_fb_df = official_fb_data[official_fb_data['Page name']==fb_post_filter]
        filter_current_week_insta_df = official_insta_data[official_insta_data['Account name']==insta_post_filter]
        #previous week filter
        filter_previous_week_fb_df = official_previous_fb_data[official_previous_fb_data['Page name']==fb_post_filter]
        filter_previous_week_insta_df = official_previous_insta_data[official_previous_insta_data['Account name']==insta_post_filter]
        # Only Varahe Post
        varahe_fb_table = filter_current_week_fb_df[filter_current_week_fb_df['Post Owner Tag']=='Varahe']
        varahe_insta_table = filter_current_week_insta_df[filter_current_week_insta_df['Post Owner Tag']=='Varahe']
        # Only Party Post
        party_fb_table = filter_current_week_fb_df[filter_current_week_fb_df['Post Owner Tag']=='Party']
        party_insta_table = filter_current_week_insta_df[filter_current_week_insta_df['Post Owner Tag']=='Party']
        # Facebook stats list
        current_week_fb_stats = post_stats(filter_current_week_fb_df,'Post ID','Permalink','Likes','Comments','Shares','People reached','Engagements','Impressions')
        previous_week_fb_stats = post_stats(filter_previous_week_fb_df,'Post ID','Permalink','Likes','Comments','Shares','People reached','Engagements','Impressions')
        # Insta Stats list
        current_week_insta_stats = post_stats(filter_current_week_insta_df,'Post ID','Permalink','Likes','Comments','Shares','Reach','Engagements','Impressions')
        previous_week_insta_stats = post_stats(filter_previous_week_insta_df,'Post ID','Permalink','Likes','Comments','Shares','Reach','Engagements','Impressions')
        #top bottom contents 
        varahe_fb_top_bottom_contents = top_bottom_posts(varahe_fb_table,'Post ID','People reached','Post type','Permalink','Likes','Comments','Shares','Engagements','Impressions','Description')
        varahe_insta_top_bottom_contents = top_bottom_posts(varahe_insta_table,'Post ID','Reach','Post type','Permalink','Likes','Comments','Shares','Engagements','Impressions','Description')
        #Content table
        varahe_fb_content = content_performance(varahe_fb_table,'Post ID','Permalink','Likes','Comments','Shares','People reached','Engagements','Impressions')
        varahe_insta_content = content_performance(varahe_insta_table,'Post ID','Permalink','Likes','Comments','Shares','Reach','Engagements','Impressions')
        party_fb_content = content_performance(party_fb_table,'Post ID','Permalink','Likes','Comments','Shares','People reached','Engagements','Impressions')
        party_insta_content = content_performance(party_insta_table,'Post ID','Permalink','Likes','Comments','Shares','Reach','Engagements','Impressions')
        #makeing horizontal
        fb_post_stats = current_week_fb_stats + previous_week_fb_stats
        insta_post_stats = current_week_insta_stats + previous_week_insta_stats
        fb_stats_df.loc[len(fb_stats_df)] = fb_post_stats
        insta_stats_df.loc[len(insta_stats_df)] = insta_post_stats
        fb_top_bottom_df.loc[len(fb_top_bottom_df)] = varahe_fb_top_bottom_contents
        insta_top_bottom_df.loc[len(insta_top_bottom_df)] = varahe_insta_top_bottom_contents
        initial_df.loc[len(initial_df)] = initial_list
        fb_content_list = varahe_fb_content + party_fb_content
        fb_content_performance_df.loc[len(fb_content_performance_df)] = fb_content_list
        insta_content_list = varahe_insta_content + party_insta_content
        insta_content_performance_df.loc[len(insta_content_performance_df)] = insta_content_list
    result_df = pd.concat([initial_df,fb_stats_df,insta_stats_df,fb_content_performance_df,insta_content_performance_df,fb_top_bottom_df,insta_top_bottom_df],axis=1)
    calculate_growth_with_conditions(result_df,'current_week_fb_total_post','previous_week_fb_total_post','fb_post_growth')
    calculate_growth_with_conditions(result_df,'current_week_fb_total_reach','previous_week_fb_total_reach','fb_reach_growth')
    calculate_growth_with_conditions(result_df,'current_week_fb_total_engagement','previous_week_fb_total_engagement','fb_engagement_growth')
    calculate_growth_with_conditions(result_df,'current_week_insta_total_post','previous_week_insta_total_post','insta_post_growth')
    calculate_growth_with_conditions(result_df,'current_week_insta_total_reach','previous_week_insta_total_reach','insta_reach_growth')
    calculate_growth_with_conditions(result_df,'current_week_insta_total_engagement','previous_week_insta_total_engagement','insta_engagement_growth')
    #result_df['fb_post_growth'] = (((result_df['current_week_fb_total_post'] - result_df['previous_week_fb_total_post']) / result_df['previous_week_fb_total_post'] )* 100).astype(int).astype(str) + '%'
    result_df['Created'] = ''
    result_df['Date'] =''
    result_df.fillna('',inplace=True)
    result_sheet = client.open_by_url(posting_sheet_link)
    #created json file of all the entries
    result_sheet_data = result_sheet.worksheet(posting_sheet_name)
    # convert the json to dataframe
    ss = result_sheet_data.update([result_df.columns.values.tolist()] + result_df.values.tolist())
    st.write("Offical Smaar Data is Updated")
    if ss:        
        posting_sheet = client.open_by_url(posting_sheet_link)
        dd = create_weekly_sheet_copy(client, posting_sheet, posting_sheet_name)  
        if dd:
            script_button = st.button('Run Script')
            if script_button:
                scripting(creddd)
                            
if __name__ == "__main__":
    main()    
