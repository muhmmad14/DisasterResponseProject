import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    Load messages and categories datasets and merge using common id function.

    Parameters:
        messages_filepath: csv path of file containing messages.
        categories_filepath: csv path of file containing categories.

    Returns:
        df: combined dataset of messages and categories.   
    '''
    
    # loading datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv (categories_filepath)
    
    # merging the two datasets
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    
    """
    Clean categories data function.
    Arguments:
        df -> combined dataset of messages and categories
    Output:
        df -> combined dataset of messages and categories cleaned
    """
    
    # spliting the categories to columns
    categories_split = df['categories'].str.split(pat = ';', expand = True)
    
    # renaming the categories columns names
    row = categories_split.iloc[0]
    category_colnames = row.apply(lambda x: x.rstrip('- 0 1'))
    categories_split.columns = category_colnames
    
    # change the values to (0 or 1)
    for column in categories_split:
        categories_split[column] = categories_split[column].str[-1]
        categories_split[column] = pd.to_numeric(categories_split[column], errors = 'coerce')
    
    
    # change the categories columns in df dataset
    df.drop(['categories'], axis = 1, inplace = True)
    df = pd.concat([df, categories_split], axis = 1, sort = False)
    
    df.drop_duplicates(inplace=True)
    df = df.drop(['child_alone'],axis=1)
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    return df


def save_data(df, database_filename):
    
    """
    Save the clean dataset into an sqlite database function.
    Arguments:
        df -> combined dataset of messages and categories cleaned
        database_filename -> path to SQLite database
    """ 
    
    engine = create_engine('sqlite:///'+ str (database_filename))
    df.to_sql('MessagesCategories', engine, index=False, if_exists = 'replace')


def main():
    
    """
    Main function that execute the data processing functions. There are three primary actions taken by this function:
        1) Load messages and categories datasets and merge them
        2) Clean categories data
        3) Save the clean dataset into an sqlite database function
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()