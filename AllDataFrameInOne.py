import pandas as pd
import util


class AllDataFrameInOne:
    def __init__(self, rentDataFile: str, crimeDataFile: str):
        self._rent_df_ = util.format_read_csv(rentDataFile)
        self._borough_crime_df_ = util.format_read_csv(crimeDataFile)
        # The df used for linear regression
        self.df_reg_ = None

    def filter_crime_minor(self, tar_col: str):
        return self._borough_crime_df_[self._borough_crime_df_['MinorText'] == tar_col]

    def join_tables(self, crime_type: str, crime_year_: int, rent_year: str):
        """
        @TODO: Comments on the way...
        :param crime_type: The specific type of crime in 'MinorText' column
        :param crime_year_: The first year sum from
        :param rent_year: In type of '20XX-XX'
        :return: A left joint dataset contains borough name, crime data sum in a year and rent year
        """
        filtered_crime_df = self.filter_crime_minor(crime_type)
        columns_2020 = [str(year_month) for year_month in range(crime_year_, crime_year_+12)]
        filtered_crime_df['year_total'] = filtered_crime_df[columns_2020].sum(axis=1)
        filtered_crime_df = filtered_crime_df.iloc[:, :3].join(filtered_crime_df['year_total'])

        joined_data = pd.merge(filtered_crime_df, self._rent_df_,
                               left_on='LookUp_BoroughName',
                               right_on='Area', how='left')

        # Handle the NA data and remove them
        joined_data.replace('..', pd.NA, inplace=True)
        joined_data.dropna(inplace=True)
        self.df_reg_ = joined_data[['Area', 'year_total', rent_year]]
        return self.df_reg_

    def print_column_names(self):
        print(self._rent_df_.columns)
        print("____")
        print(self._borough_crime_df_.columns)

