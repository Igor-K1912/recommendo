import pandas as pd
import numpy as np
import joblib
import re
import os
import random

def create_map(df, columns):
    map_result = {}
    for item in columns:
        val = df[df['name'] == item]['displayname'].values
        map_result[item] = val[0] if val.size > 0 and val[0] == val[0]  else item  # check nan

    return map_result


# Set pandas option to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

folder_path = ''

# Read column names from the CSV file
column_names = pd.read_csv(folder_path + 'samples_merged_df.csv', nrows=0).columns
columns_to_convert = [col for col in column_names if col.startswith(('i_', 'cm_'))]
dtype_dict = {col: 'int8' for col in columns_to_convert}
dtype_dict['ic_site_id'] = 'uint16'
dtype_dict['CTR'] = 'float16'
#dtype_dict['iCTR'] = 'float16'
# Read the CSV file with the specified dtypes

print('Loading data ... ')
samples_df = pd.read_csv(folder_path + 'samples_merged_df.csv', dtype=dtype_dict)
cat_columns = samples_df.filter(regex=f'^i_').columns
segm_columns = samples_df.filter(regex=f'^cm_').columns
all_keys_values_df = pd.read_csv(folder_path + 'key_value.csv')
filtered_all_keys_values_df = all_keys_values_df[all_keys_values_df['name'].notna() & all_keys_values_df['name'].str.startswith(('i_', 'cm_'))]
site_id_domain_df = pd.read_csv(folder_path + 'map_icsiteid_domain.csv')
map_name_display_name = create_map(filtered_all_keys_values_df, cat_columns) | create_map(filtered_all_keys_values_df, segm_columns)
map_siteid_domain = dict(zip(site_id_domain_df['ic_site_id'], site_id_domain_df['url_domain']))
samples_df['ic_site_id'] = samples_df['ic_site_id'].map(map_siteid_domain).fillna('Unknown')
samples_df.rename(columns={'ic_site_id': 'domain_url'}, inplace=True)


class CustomEx(Exception):
    def __init__(self, msg):
        self.msg = msg
        self.filtered_df = None
        super().__init__(self.msg)

class Statistics:
    def convert_name_to_displayname(self, df, cols):
        df.rename(columns=cols, inplace=True)

    def custom_print(self, msg):
        print(f"{'*' * 150}\n{msg}\n{'*' * 150}\n")

    def print_table_header(self, feature_name, header, additional_header=''):
        """Helper function for common table printing logic"""
        print('\n')
        print(f"{header}")
        print("=" * 150)
        print(f"{feature_name.upper():<50} {additional_header}")
        print('-' * 150)

    def print_ctr(self, feature_name, ctr, average_ctr, main_feature='advertiser', performance_index_included=False):
        """Print top features by CTR"""
        self.print_table_header(feature_name, f'Top {feature_name} by CTR (regardless of other features). Average CTR means average over all features for the current {main_feature} (seasons, sites, etc)', f"{'<CTR>':>8} {'Average CTR':>15} {'Performance Index':>20}")
        for i, (feature, feature_ctr) in enumerate(ctr[:9], 1):
            performance_index = (feature_ctr - average_ctr) / average_ctr * 100 if average_ctr > 0 else 0.0
            if performance_index_included:
                performance_index = feature_ctr[1]
                feature_ctr = feature_ctr[0]
            print(f"{i}. {str(feature):<50} {feature_ctr:<8.4f} {average_ctr: >10.4}  {performance_index: >12.4}")

    def print_top_influence(self, feature_name, coef_influence):
        """Print influence of features on CTR"""
        self.print_table_header(feature_name, f'Influence of {feature_name} on the CTR (regardless of other features)', 'influence (negative values mean that activating these segments tends to reduce CTR))')
        for i, (feature, influence) in enumerate(coef_influence[:9], 1):
            print(f"{i}. {feature:<50} {influence:>6.4f}")

    # Best performing feature values
    def top_single_feature_st(self, feature):
        return list(self.filtered_df.groupby(feature)['CTR'].mean().sort_values(ascending=False).items())


class Adv_Spec_Stat(Statistics):

    def __init__(self, advertiserid):
        self.advertiser_df = pd.read_csv(folder_path + 'advertiser.csv')
        self.advertiserid = advertiserid
        self.filtered_df = samples_df[samples_df['advertiserid'] == self.advertiserid]

        # Initializing of variables
        self.top_highest_ctr_df = pd.DataFrame()
        self.top_segm = None
        self.top_cat = None
        self.average_ctr = None
        self.top_cat_influence = None
        self.top_segm_influence = None
        self.top_ad_place = None
        self.top_sites = None
        self.top_season = None

    def print_(self):
        self.custom_print(f'Top {self.top_highest_ctr_df.shape[0]} combinations')
        print(self.top_highest_ctr_df)
        self.print_ctr('segment', self.top_segm, self.average_ctr)
        self.print_ctr('category', self.top_cat, self.average_ctr)
        self.print_top_influence('segments', self.top_segm_influence)
        self.print_top_influence('categories', self.top_cat_influence)
        self.print_ctr('ad placement', self.top_ad_place, self.average_ctr)
        self.print_ctr('sites', self.top_sites, self.average_ctr)
        self.print_ctr('season', self.top_season, self.average_ctr)

    def run_pipeline(self):
        name, num = self.check_advid()
        self.custom_print(f'STATISTICAL ANALYSIS OF THE EXISTING DATASET FOR THE ADVERTISER - "{name}" (YEAR 2024)')
        print('Total number of samples associated with this advertiser in the the dataset: ' , num)
        self.top_combinations()
        self.top_segm_cat()
        self.top_influence()
        self.top_ad_place = self.top_single_feature_st('bottom_level_ad_unit_name')
        self.top_sites = self.top_single_feature_st('domain_url')
        self.top_season = self.top_single_feature_st('season')

        self.print_()

    def check_advid(self):
        if self.filtered_df.empty:
            raise CustomEx('No advertiser id in the dataset')

        sampl_df = self.advertiser_df[self.advertiser_df['advertiserid'] == self.advertiserid]

        return sampl_df.advertisername.iloc[0], self.filtered_df.shape[0]


    # Best feature combinations with the highest CTR
    def top_combinations(self, count=20):
        #columns = samples_df.columns[samples_df.columns != 'CTR'].tolist()
        columns = samples_df.columns[~samples_df.columns.isin(['CTR', 'iCTR'])].tolist()
        #grouped_df = self.filtered_df.loc[self.filtered_df.groupby(columns)['CTR'].idxmax()].sort_values(by='CTR', ascending=False)
        grouped_df = self.filtered_df.loc[self.filtered_df.groupby(columns, observed=True)['CTR'].idxmax()].sort_values(by='CTR', ascending=False)
        self.top_highest_ctr_df = grouped_df.head(count)
        # Remove columns associated with sect/cat if the last ones are not specified
        #columns_to_check = self.top_highest_ctr_df.columns.difference(['CTR']).to_list()
        self.top_highest_ctr_df = self.top_highest_ctr_df.loc[:, (self.top_highest_ctr_df != 0).any(axis=0) |  pd.Series(self.top_highest_ctr_df.columns == 'CTR', index=self.top_highest_ctr_df.columns)]
        self.convert_name_to_displayname(self.top_highest_ctr_df, map_name_display_name)


    # Mean CTR for each segment/category regardless of other features
    def top_segm_cat(self):

        def top_feature(cols):
            feature = {}
            for col in cols:
                mean_ctr = self.filtered_df[self.filtered_df[col] == 1]['CTR'].mean()
                if mean_ctr == mean_ctr:  # check nan
                    feature[re.sub(r'^(i_|cm_)', '', map_name_display_name[col])] = mean_ctr

            return sorted(feature.items(), key=lambda x: x[1], reverse=True)

        # Average CTR for the advertiser
        self.average_ctr = self.filtered_df['CTR'].mean()

        self.top_cat = top_feature(cat_columns)
        self.top_segm = top_feature(segm_columns)



    # ---------- Important Notes !!! -------------

    # The approach above only estimates the marginal impact of each individual segment/category, but does not consider potential interactions
    # between categories. If categories frequently appear together (i.e., certain categories are active together in many rows), their impact
    # may be different when considering their combinations. To handle interactions (including logical relations such as "IS" or "IS_NOT"),
    # we need to consider combinations of categories and use a recommendation model that can automatically consider interactions between features.

    # Let's estimate the influence of each individual segment/category on the CTR, even though the combination of categories (interactions between them)
    # can be important. To estimate the influence of individual categories, we can:

    #  > Calculate the mean CTR for each segment/category when it is 1 (included in custom targeting field), while ensuring the other categories are either included or excluded.
    #  > Compare the mean CTR of each category (when it is 1) with the mean CTR when it is 0 or -1 (not specified/excluded).

    # It should be noted that in the case of a large data set, we will most likely get the same result as in the case of the simple analysis performed above.

    def top_influence(self):

        def top_feature(cols):
            feature_influence = {}

            for col in cols:
                # Mean CTR when the segment/category is active
                ctr_active = self.filtered_df[self.filtered_df[col] == 1]['CTR'].mean()

                # Mean CTR when the segment/category is inactive/disable
                ctr_inactive = self.filtered_df[self.filtered_df[col] < 1]['CTR'].mean()

                if pd.isna(ctr_active) or pd.isna(ctr_inactive):
                    continue

                feature_influence[re.sub(r'^(i_|cm_)', '', map_name_display_name[col])] = ctr_active - ctr_inactive

            return sorted(feature_influence.items(), key=lambda x: x[1], reverse=True)

        self.top_cat_influence = top_feature(cat_columns)
        self.top_segm_influence = top_feature(segm_columns)



class Recommendo_Base(Statistics):
    def __init__(self, df, feature):
        self.site_df = self.create_set_comb_df('ic_site_id')
        self.season_df = self.create_set_comb_df('season')
        self.bottom_level_ad_unit_name_df = self.create_set_comb_df('bottom_level_ad_unit_name')
        self.height_width_df = self.create_set_comb_df('height_width')
        self.segm_df = self.add_zero_raw(self.create_set_comb_df('cm'))
        self.cat_df = df if feature == 'category' else self.add_zero_raw(self.create_set_comb_df('i'))
        self.adv_lineitemtype_df = df if feature == 'advertiser' else self.create_set_comb_df('lineitemtype')

        self.columns = None
        self.map_feature_coef = {}
        self.chunk_size = 2000

        self.map_feature = {
            'segment': ('i_segm', self.segm_df),
            'category': ('i_cat', self.cat_df),
            'size': ('i_size', self.height_width_df),
            'placement': ('i_ad_unit', self.bottom_level_ad_unit_name_df),
            'season': ('i_season', self.season_df),
            'site': ('i_site', self.site_df),
            'lineitemtype': ('i_adv_or_lineitemtype', self.adv_lineitemtype_df),
            'bottom_level_ad_unit_name': ('i_ad_unit', self.bottom_level_ad_unit_name_df)
        }

        self.sizes = [ # order matters !!!
            self.height_width_df.shape[0],
            self.season_df.shape[0],
            self.bottom_level_ad_unit_name_df.shape[0],
            self.segm_df.shape[0],
            self.cat_df.shape[0],
            self.site_df.shape[0],
            self.adv_lineitemtype_df.shape[0]
        ]

        # should be the same order as for `self.sizes`
        self.feature_columns = ['i_size', 'i_season', 'i_ad_unit', 'i_segm', 'i_cat', 'i_site', 'i_adv_or_lineitemtype']

        # Get the number of available cores and reduce by 1 (just to let the system be alive ;))
        self.n_cors = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

    def check_input(self, inp, prefix):

        if not inp:
            return

        inp_modified = [prefix + '_' + str(i) for i in inp]

        items = [item for item in self.columns if item.startswith(f'{prefix}_')]

        set1 = set(inp_modified)
        set2 = set(items)

        elements_outside_intersection = set1 - set1.intersection(set2)

        if elements_outside_intersection:
            print(f'There is no elements with this ID in the dataset: {elements_outside_intersection}')
            print(f'List of available elements: {set2}')
            return ([], prefix)

        return inp, prefix

    def predict_for_combinations(self, combinations):
        # pd -> numpy
        # just to speed up their processing inside the loop
        segm_np = self.segm_df.to_numpy()
        cat_np = self.cat_df.to_numpy()
        bottom_level_np = self.bottom_level_ad_unit_name_df.to_numpy()
        site_np = self.site_df.to_numpy()
        season_np = self.season_df.to_numpy()
        height_width_np = self.height_width_df.to_numpy()
        adv_lineitemtype_np = self.adv_lineitemtype_df.to_numpy()

        # Iterate through all combinations and collect the feature vectors
        feature_vectors = [
            np.concatenate([
                segm_np[i_segm],
                cat_np[i_cat],
                adv_lineitemtype_np[i_adv_or_lineitemtype],
                bottom_level_np[i_ad_unit],
                site_np[i_site],
                season_np[i_season],
                height_width_np[i_size]
            ])
            for (i_size, i_season, i_ad_unit, i_segm, i_cat, i_site, i_adv_or_lineitemtype) in combinations
        ]

        # Construct the DataFrame once
        df_temp = pd.DataFrame(
            feature_vectors,
            columns=(
                    list(self.segm_df.columns) +
                    list(self.cat_df.columns) +
                    list(self.adv_lineitemtype_df.columns) +
                    list(self.bottom_level_ad_unit_name_df.columns) +
                    list(self.site_df.columns) +
                    list(self.season_df.columns) +
                    list(self.height_width_df.columns)
            )
        )

        # Predict using the model
        predictions = self.model.predict(df_temp)

        # Combine predictions with keys
        new_rows = [
            (*combinations[idx], prediction) for idx, prediction in enumerate(predictions)
        ]


        return new_rows

    def load_decoded_columns(self, filename):
        with open(filename, 'r') as file:
            decoded_columns = file.read().strip().split(',')

        return decoded_columns

    def create_set_comb_df(self, column_prefix):
        df = pd.DataFrame([[0] * len(self.columns)], columns=self.columns)
        cols = df.filter(regex=f'^{column_prefix}_').columns

        return pd.DataFrame(np.eye(len(cols)), columns=cols, dtype='int8')

    def set_value(self, values, column_prefix):
        df = pd.DataFrame([[0] * len(self.columns)], columns=self.columns)
        cols = df.filter(regex=f'^{column_prefix}_').columns
        temp_df = pd.DataFrame([[0] * len(cols)], columns=cols, dtype='int8')

        if not values:
            return

        for val in values:
            temp_df[f'{column_prefix}_{val}'] = 1

        return temp_df

    def add_zero_raw(self, df):
        if df.shape[0] > 1:
            row_with_zeros = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
            df = pd.concat([df, row_with_zeros], ignore_index=True)

        return df

    # Initialize a random population
    def initialize_population(self, popul_size, locked_bit=None):
        np.random.seed()
        # Generate pop_size random combinations
        combinations = [
            [np.random.randint(0, size) for size in self.sizes]
            for _ in range(popul_size)
        ]

        # Even distribution among the samples
        if locked_bit is not None:
            bit = 0
            for i in range(popul_size):
                combinations[i][locked_bit] = bit
                bit = bit + 1 if bit < self.sizes[locked_bit] - 1 else 0

        return combinations

    def split_combinations(self, popul_size, combinations):
        for i in range(0, popul_size, self.chunk_size):
            yield combinations[i:i + self.chunk_size]

    def selection(self, population_with_scores, tournament_size=3):
        selected = []
        for _ in range(len(population_with_scores)):
            tournament = random.sample(population_with_scores, tournament_size)
            winner = max(tournament, key=lambda x: x[7])[0:7]
            selected.append(winner)
        return selected

    # Crossover function
    def crossover(self, crossover_rate, parent1, parent2):
        # Create two child individuals by randomly selecting genes from either parent
        if np.random.random() < crossover_rate:
            # child1 = tuple(np.random.choice([p1, p2]) for p1, p2 in zip(parent1, parent2))
            # child2 = tuple(np.random.choice([p2, p1]) for p1, p2 in zip(parent1, parent2))
            crossover_point = np.random.randint(1, len(parent1))  # Select a crossover point
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2

    def mutation(self, mutation_rate, individual):
        individual = list(individual)  # Convert individual to a list to mutate in-place

        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                # Mutate by a random integer in the range of -1 to 1
                mutation_amount = np.random.randint(-1, 2)  # Random value: -1, 0, or 1

                individual[i] += mutation_amount
                # Ensure the individual stays within bounds (integer values)
                individual[i] = max(min(individual[i], self.sizes[i] - 1), 0)

        return tuple(individual)

    # Genetic algorithm
    def genetic_algorithm(self):

        # Define the population size and number of generations
        popul_size = self.chunk_size * 100
        num_generations = 50
        mutation_rate = 0.8
        crossover_rate = 1
        top_n = min(self.chunk_size, popul_size)  # Number of top combinations to track

        population = self.initialize_population(popul_size)

        best_fitness = 0

        for generation in range(num_generations):

            # Evaluate fitness for all individuals in the population
            results = joblib.Parallel(n_jobs=self.n_cors, verbose=10)(
                joblib.delayed(self.predict_for_combinations)(chunk) for chunk in self.split_combinations(popul_size, population)
            )

            # results = []
            # for chunk in self.split_combinations(popul_size, population):
            #     result = self.predict_for_combinations(chunk)
            #     results.append(result)

            population_with_scores = [tuple(row) for result in results for row in result]

            # Sort by fitness in descending order, keeping the top N combinations
            population_with_scores.sort(key=lambda x: x[7], reverse=True)

            best_fitness_back = best_fitness

            best_fitness = population_with_scores[0][7]

            if best_fitness <= best_fitness_back:
                break

            # Update the top N combinations list (only keep top N)
            top_combinations = population_with_scores[: top_n]

            best_combination = population_with_scores[0][0:7]  # 7 features

            population = self.selection(population_with_scores, 10)

            next_population = []
            for i in range(0, len(population) - 1, 2):
                parent1, parent2 = population[i], population[i + 1]

                child1, child2 = self.crossover(crossover_rate, parent1, parent2)

                next_population.append(self.mutation(mutation_rate, child1))
                next_population.append(self.mutation(mutation_rate, child2))

            # Replace the old population with the new one, preserving the best individual
            next_population[0] = best_combination

            population = next_population

            #print('Generation:', generation, '  Best iCTR:', best_fitness)
            print('Generation:', generation, '  Best CTR:', best_fitness)

        # Return the top N combinations (sorted by fitness)
        return top_combinations

    def get_unique_comb(self, top_combinations):
        top_combinations_unique = []
        seen = set()

        for individual in top_combinations:
            if (t := tuple(individual[0:7])) not in seen:
                top_combinations_unique.append(individual)
                seen.add(t)

        return top_combinations_unique


    def remove_prefix(self, d, prefix):
        if isinstance(d, pd.Series):  # If the input is a pandas Series
            return d.str.replace(rf'^{prefix}', '', regex=True)
        elif isinstance(d, pd.DataFrame):  # If the input is a pandas DataFrame
            return d.apply(lambda col: col.str.replace(rf'^{prefix}', '', regex=True) if col.dtype == 'object' else col)
        else:
            raise ValueError("Input must be a pandas Series or DataFrame")


    def split_bins(self, ctr_column):
        # Define the bins (ranges)
        #step = 1  # iCTR
        # bins = np.arange(0, 1000, step)  # Define the ranges: 0-0.1, 0.1-0.2, etc.
        step = 0.2
        bins = np.arange(0, 10, step)  # Define the ranges: 0-0.1, 0.1-0.2, etc.
        labels = [f'{round(i, 2)}-{round(i + step, 2)}' for i in bins[:-1]]  # Labels for the ranges

        return pd.cut(ctr_column, bins=bins, labels=labels, right=False)


    def get_top_comb(self, r_df, features, n_top):
        # Sort r_df and extract top 5 rows based on prediction
        temp_df = r_df.sort_values('prediction', ascending=False).head(n_top)
        prediction = temp_df['prediction']

        top_columns = []

        for feature in features:
            feature_indices = temp_df[self.map_feature[feature][0]]
            feature_top_df = self.map_feature[feature][1].loc[feature_indices]
            column_values_df = feature_top_df.apply(lambda row: '-' if (row == 0).all() else row.idxmax(), axis=1)  # set '-' if no good options for category/segment
            column_values_df = self.remove_prefix(column_values_df, "(height_width_|bottom_level_ad_unit_name_|season_|ic_site_id_)")
            top_columns.append(column_values_df.reset_index(drop=True))

        # Combine the feature columns and the predictions into a final DataFrame
        top = pd.DataFrame({
            **{feature: col for feature, col in zip(features, top_columns)},
            #'iCTR': prediction.reset_index(drop=True)
            'CTR': prediction.reset_index(drop=True)
        })

        top['site'] = top['site'].astype('int16').map(map_siteid_domain).fillna('Unknown')
        top['segment'] = top['segment'].map(map_name_display_name).fillna('Not specified')
        top['category'] = top['category'].map(map_name_display_name).fillna('Not specified')
        top = self.remove_prefix(top, "(cm_|i_)")

        # top['iCTR_range'] = self.split_bins(top['iCTR'])
        # top.drop(columns=['iCTR'], inplace=True)

        top['CTR_range'] = self.split_bins(top['CTR'])
        top.drop(columns=['CTR'], inplace=True)

        return top


    def run(self):
        top_combinations = self.genetic_algorithm()
        top_combinations_unique = self.get_unique_comb(top_combinations)
        result_df = pd.DataFrame(top_combinations_unique, columns=self.feature_columns + ['prediction'])
        result_df = result_df.astype({col: 'int16' for col in self.feature_columns})

        return self.get_top_comb(result_df, ['segment', 'category', 'size', 'placement', 'season', 'site'], min(50, result_df.shape[0]))


    def predict_top_single_feature(self, popul_size, locked_bit):

        population = self.initialize_population(popul_size, locked_bit)

        results = joblib.Parallel(n_jobs=self.n_cors, verbose=10)(
            joblib.delayed(self.predict_for_combinations)(chunk) for chunk in self.split_combinations(popul_size, population)
        )

        # results = []
        # for chunk in self.split_combinations(popul_size, population):
        #     result = self.predict_for_combinations(chunk)
        #     results.append(result)

        return [tuple(row) for result in results for row in result]

    def top_single_feature(self, coef, feature_name):

        # Function to calculate the performance index
        def calculate_performance_index(val, avg):
            # aver_in_range = sum(map(float, val['iCTR_range'].split('-')))/2
            # diff = aver_in_range - avg if abs(aver_in_range - avg) >= 0.5 else 0  # depends on step !!!

            aver_in_range = sum(map(float, val['CTR_range'].split('-'))) / 2
            diff = aver_in_range - avg if abs(aver_in_range - avg) >= 0.1 else 0  # depends on step !!!

            return (diff / aver_in_range * 100) if aver_in_range != 0 else 0

        # locked_bit
        bits = {
            'size': 0,
            'season': 1,
            'bottom_level_ad_unit_name': 2,
            'segment': 3,
            'category': 4,
            'site': 5,
            'lineitemtype': 6
        }

        popul_size  = coef * self.chunk_size

        locked_bit = bits[feature_name]
        all_combinations = self.predict_top_single_feature(popul_size, locked_bit)
        result_df = pd.DataFrame(all_combinations, columns=self.feature_columns + ['prediction'])
        result_df = result_df.astype({col: 'int16' for col in self.feature_columns})
        result_df['prediction'] = result_df['prediction'].apply(lambda x: 0 if x < 0 else x)

        mean_ctr = result_df.groupby(self.feature_columns[locked_bit])['prediction'].mean()
        temp_df = self.map_feature[feature_name][1].copy()
        # temp_df['iCTR_mean'] = mean_ctr
        temp_df['CTR_mean'] = mean_ctr

        columns_with_ones = temp_df.apply(lambda row: row.index[row == 1].any(), axis=1)
        # result_df = pd.concat([columns_with_ones, temp_df['iCTR_mean']], axis=1)
        # result_df.columns = [feature_name, 'iCTR_mean']
        result_df = pd.concat([columns_with_ones, temp_df['CTR_mean']], axis=1)
        result_df.columns = [feature_name, 'CTR_mean']

        prefixes = {
            'segment': "cm_",
            'category': "i_",
            'size': 'height_width_',
            'bottom_level_ad_unit_name': 'bottom_level_ad_unit_name_',
            'site': 'ic_site_id_',
            'season': 'season_',
            'lineitemtype': 'lineitemtype_'
        }

        if feature_name in ['segment', 'category']:
            result_df[feature_name] = result_df[feature_name].map(map_name_display_name).fillna('Not specified')

        result_df = self.remove_prefix(result_df, prefixes[feature_name])

        if feature_name == 'site':
            result_df[feature_name] = result_df[feature_name].astype('int16').map(map_siteid_domain).fillna('Not specified')

        # Create a new column 'CTR_range' by binning the 'CTR' values
        #result_df['iCTR_range'] = self.split_bins(result_df['iCTR_mean'])
        result_df['CTR_range'] = self.split_bins(result_df['CTR_mean'])

        # Group by 'CTR_range' and collect the segments into a list for each range
        #grouped_df = result_df.groupby('iCTR_range', observed=False)[feature_name].apply(list).reset_index()
        grouped_df = result_df.groupby('CTR_range', observed=False)[feature_name].apply(list).reset_index()

        # filter out any groups where the list of feature_name values is empty
        grouped_df = grouped_df[grouped_df[feature_name].apply(lambda x: len(x) > 0)].reset_index(drop=True)

        grouped_df['Performance'] = grouped_df.apply(lambda row: calculate_performance_index(row, mean_ctr.mean()), axis=1)

        #return grouped_df.sort_values(by='iCTR_range', ascending=False)
        return grouped_df.sort_values(by='CTR_range', ascending=False)

    def run_pipeline(self):
        while 1:
            print('\nWhat are you interested in:')
            for num, f in enumerate(self.map_feature_coef.keys()):
                print(f'{"placement" if num == 4 else f}: {num}')

            print('Best combinations (may take up from 10 to 20 minutes): 6')

            i = int(input('Enter number (0-6) or 7 to exit: ') or 7)

            if i == 7:
                break
            elif i == 6:
                print(self.run())
                continue

            feature = list(self.map_feature_coef.keys())[i]
            get_top_feature = self.top_single_feature(self.map_feature_coef[feature], feature)
            #  Print results
            print('\n')
            #self.custom_print('Impression-Weighted Click-Through Rate (iCTR) for providing a balanced view: iCTR=CTR*log(Impressions)')
            # self.custom_print(f"{'Predict.iCTR Range':<22} {'Performance (%)':<20} {feature}")
            # for index, row in get_top_feature.iterrows():
            #     print(f"{row['iCTR_range']:<22} {row['Performance']:<20.2f} {row[feature]}")

            self.custom_print(f"{'Predict.CTR Range':<22} {'Performance (%)':<20} {feature}")
            for index, row in get_top_feature.iterrows():
                print(f"{row['CTR_range']:<22} {row['Performance']:<20.2f} {row[feature]}")

            print('\n')


class Recommendo_Adv(Recommendo_Base):
    def __init__(self, advertiserid):
        self.model = joblib.load('model_with_advertiser.pkl')
        self.columns = self.load_decoded_columns('features_adv.txt')
        df = self.set_value(*self.check_input([advertiserid], 'advertiserid'))

        super().__init__(df, 'advertiser')

        self.map_feature_coef = {
            'segment': 40,
            'category': 60,
            'size': 20,
            'season': 10,
            'bottom_level_ad_unit_name': 10,
            'site': 40,
        }


class Cat_Level_Spec_Stat(Statistics):
    def __init__(self, cat_name):
        self.category_name = cat_name
        self.average_ctr = None
        self.filtered_df = None

    def run_pipeline(self):
        self.custom_print(f'STATISTICAL ANALYSIS OF THE EXISTING DATASET FOR THE CATEGORY - "{self.check_category()}" (YEAR 2024)')
        self.filtered_df = samples_df[samples_df[self.category_name] == 1]
        count_non_zero = samples_df[samples_df[self.category_name].isin([1, -1])].shape[0]
        # Average across all segments (regardless of the other features)
        self.average_ctr = self.filtered_df[self.filtered_df[segm_columns].any(axis=1) == 1]['CTR'].mean()
        print('Total number of samples associated with this category in the dataset: ', count_non_zero)
        self.print_()


    def check_category(self):
        if self.category_name not in cat_columns:
            raise CustomEx('This category is not in the dataset.')

        return map_name_display_name[self.category_name].replace('i_', '')

    def get_top_segm(self):
        feature = {}
        for col in segm_columns:
            mean_ctr = self.filtered_df[self.filtered_df[col] == 1]['CTR'].mean()
            if mean_ctr == mean_ctr:  # check for nan
                feature[map_name_display_name.get(col, 'Unknown')] = mean_ctr

        return sorted(feature.items(), key=lambda x: x[1], reverse=True)

    def get_seasonal_info(self):
        feature = {}

        for season in ['winter', 'spring', 'summer', 'fall']:
            for col_segm in segm_columns:
                season_df = self.filtered_df[(self.filtered_df[col_segm] == 1) & (self.filtered_df['season'] == season)]
                mean_ctr = season_df['CTR'].mean()
                if pd.notna(mean_ctr):
                    performance = ((mean_ctr - self.average_ctr) / self.average_ctr) * 100
                    feature[(season, map_name_display_name.get(col_segm, 'Unknown'))] = (mean_ctr, performance)

        return sorted(feature.items(), key=lambda x: x[1], reverse=True)


    def print_(self):
        self.print_ctr('segment', self.get_top_segm(), self.average_ctr, 'category')
        self.print_ctr('season/segment', self.get_seasonal_info(), self.average_ctr, 'category', True)
        self.print_ctr('sites', self.top_single_feature_st('domain_url'), self.average_ctr, 'category')
        self.print_ctr('ad placement', self.top_single_feature_st('bottom_level_ad_unit_name'), self.average_ctr, 'category')


class Recommendo_No_Adv(Recommendo_Base):
    def __init__(self, cat_name):
        self.model = joblib.load('model_no_advertiser.pkl')
        self.columns = self.load_decoded_columns('features.txt')
        df = self.set_value(*self.check_input([cat_name.replace('i_', '')], 'i'))
        super().__init__(df, 'category')

        self.map_feature_coef = {
            'segment': 40,
            'lineitemtype': 10,
            'size': 20,
            'season': 10,
            'bottom_level_ad_unit_name': 10,
            'site': 40,
        }


try:
    user_story = int(input("Choose the manager type (enter 1 or 2): 1 (default) - Advertising Account Manager, 2 - Category Manager: ") or 1)

    if user_story == 1:
        adv_id = int(input('Enter the advertiser id (e.g. 4855555657): '))
        stat = Adv_Spec_Stat(adv_id)
    elif user_story == 2:
        category_name = input('Enter the category name using short format i_XXX (e.g. i_559): ')
        stat = Cat_Level_Spec_Stat(category_name)
    else:
        raise CustomEx('Wrong choice. Possible options: 1 or 2')

    stat.run_pipeline()

    if input("\nDo you want to get recommendations (y/n)? default - no: ").lower() == 'y':
        recom = Recommendo_Adv(adv_id) if user_story == 1 else Recommendo_No_Adv(category_name)
        recom.run_pipeline()

except CustomEx as e:
    print(f"Custom Error: {e.msg}")
except Exception as e:
    print(e)

# Advertisers with non-zeros cat/segm field

# Adv ID:  44773522    Adv. Name:  HDM | R | Google DoubleClick Ad Exchange    Num of records:  5101    List of cat/segm:  ['cm_0067', 'cm_0134', 'i_292', 'i_307']
# Adv ID:  53029402    Adv. Name:  HDM | D | Chanel Incorporated    Num of records:  318    List of cat/segm:  ['cm_0079', 'cm_0088', 'cm_0133', 'i_155', 'i_162', 'i_177', 'i_1KXCLD', 'i_201', 'i_23', 'i_232', 'i_283', 'i_42', 'i_432', 'i_473', 'i_552', 'i_553', 'i_555', 'i_556', 'i_557', 'i_558', 'i_559', 'i_565', 'i_576', 'i_577', 'i_578', 'i_653', 'i_671', 'i_676', 'i_JLBCU7']
# Adv ID:  53336362    Adv. Name:  HDM | D | Ralph Lauren Corporation/Ralph Lauren    Num of records:  228    List of cat/segm:  ['cm_0008', 'cm_0027', 'cm_0133', 'i_201', 'i_23', 'i_231', 'i_237', 'i_383', 'i_386', 'i_389', 'i_433', 'i_437', 'i_438', 'i_560', 'i_565', 'i_576', 'i_577', 'i_578', 'i_579', 'i_316', 'i_322', 'i_453', 'i_320', 'i_315']
# Adv ID:  53654242    Adv. Name:  HDM | D | Pandora Jewelry LLC    Num of records:  288    List of cat/segm:  ['cm_0084', 'cm_0126', 'cm_0177', 'cm_0184', 'cm_0186', 'i_1KXCLD', 'i_472', 'i_476', 'i_478']
# Adv ID:  53850802    Adv. Name:  HDM | D | The Home Depot Incorporated    Num of records:  379    List of cat/segm:  ['cm_0112', 'i_137', 'i_138', 'i_274', 'i_279', 'i_283', 'i_285', 'i_441']
# Adv ID:  58716562    Adv. Name:  HDM | D | Levi Strauss & Co/Dockers    Num of records:  65    List of cat/segm:  ['cm_0008', 'i_560', 'i_576', 'i_577', 'i_578', 'i_579']
# Adv ID:  59680642    Adv. Name:  HDM | D | Benjamin Moore & Company    Num of records:  70    List of cat/segm:  ['i_112', 'i_257', 'i_274', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285', 'i_445', 'i_451']
# Adv ID:  62130682    Adv. Name:  HDM | D | Miele Appliances    Num of records:  2    List of cat/segm:  ['i_274', 'i_278', 'i_279']
# Adv ID:  65669482    Adv. Name:  HDM | D | Vera Bradley Designs    Num of records:  33    List of cat/segm:  ['cm_0008', 'cm_0095', 'i_159', 'i_160', 'i_22', 'i_557', 'i_560', 'i_576', 'i_577', 'i_578', 'i_579', 'i_665', 'i_677', 'i_78']
# Adv ID:  67254922    Adv. Name:  HDM | D | Amazon.com/Amazon Studios    Num of records:  124    List of cat/segm:  ['cm_0024']
# Adv ID:  68412442    Adv. Name:  HDM | D | Phillips-Van Heusen Corp/Tommy Hilfiger    Num of records:  61    List of cat/segm:  ['cm_0008', 'cm_0177', 'i_476', 'i_478', 'i_552', 'i_560', 'i_576', 'i_577', 'i_578', 'i_579']
# Adv ID:  71164762    Adv. Name:  HDM | D | Fossil Incorporated/Fossil Watches    Num of records:  21    List of cat/segm:  ['cm_0177', 'i_476', 'i_478']
# Adv ID:  71878522    Adv. Name:  HMIES | D | PERNOD RICARD 4129    Num of records:  292    List of cat/segm:  ['cm_0096', 'i_1', 'i_163', 'i_192', 'i_200', 'i_211', 'i_214', 'i_216', 'i_222', 'i_286']
# Adv ID:  71909722    Adv. Name:  HMIES | D | MAHOU / SAN MIGUEL 1596    Num of records:  116    List of cat/segm:  ['i_1', 'i_192', 'i_436']
# Adv ID:  72355882    Adv. Name:  HDM | D | Shiseido Cosmetics America Ltd    Num of records:  159    List of cat/segm:  ['i_123', 'i_432', 'i_52', 'i_552', 'i_553']
# Adv ID:  82223602    Adv. Name:  HDM | D | Richemont Group/Cartier    Num of records:  437    List of cat/segm:  ['cm_0027', 'cm_0151', 'cm_0161', 'i_231', 'i_237', 'i_383', 'i_386', 'i_389', 'i_433', 'i_437', 'i_438', 'i_578', 'i_316', 'i_322', 'i_453', 'i_320', 'i_315']
# Adv ID:  82965082    Adv. Name:  HDM | D | Coty Prestige/Marc Jacobs Fragrances    Num of records:  54    List of cat/segm:  ['i_432', 'i_552', 'i_553', 'i_558', 'i_560', 'i_561', 'i_576', 'i_577', 'i_578', 'i_579', 'i_JLBCU7']
# Adv ID:  83085082    Adv. Name:  HDM | D | L'Oreal Paris/Corporate    Num of records:  129    List of cat/segm:  ['cm_0119', 'i_105', 'i_115', 'i_123', 'i_125', 'i_126', 'i_127', 'i_128', 'i_129', 'i_130', 'i_139', 'i_567', 'i_583', 'i_596', 'i_676']
# Adv ID:  88441882    Adv. Name:  HMIES | D | LACTALIS 24798    Num of records:  366    List of cat/segm:  ['i_211', 'i_212', 'i_213', 'i_214', 'i_215', 'i_216', 'i_217', 'i_218', 'i_219', 'i_220', 'i_221', 'i_222']
# Adv ID:  89938162    Adv. Name:  HDM | D | The Swatch Group SA/SMH-US Incorporated/Rado    Num of records:  106    List of cat/segm:  ['i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  101624362    Adv. Name:  HDM | D | Sephora    Num of records:  195    List of cat/segm:  ['cm_0157', 'i_232', 'i_323', 'i_557', 'i_559', 'i_590', 'i_591']
# Adv ID:  124401802    Adv. Name:  HDM | D | Fossil Incorporated/Michele Watch Company    Num of records:  47    List of cat/segm:  ['cm_0177', 'i_476', 'i_478']
# Adv ID:  139138282    Adv. Name:  HMIIT | D | HERMES (MODA)    Num of records:  37    List of cat/segm:  ['i_133', 'i_389']
# Adv ID:  144914962    Adv. Name:  HDM | D | Rolex Watch USA Incorporated    Num of records:  28    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  147170122    Adv. Name:  HDM | D | Recreational Equipment Inc-REI    Num of records:  400    List of cat/segm:  ['cm_0096', 'cm_0101', 'i_221', 'i_223', 'i_224', 'i_225', 'i_227', 'i_228', 'i_229', 'i_230', 'i_231', 'i_234', 'i_235', 'i_238', 'i_492', 'i_544', 'i_549', 'i_286', 'i_320']
# Adv ID:  173499562    Adv. Name:  HDM | D | Movado Group/Movado Group    Num of records:  21    List of cat/segm:  ['cm_0063', 'cm_0106', 'i_476', 'i_478', 'i_565']
# Adv ID:  195333082    Adv. Name:  HDM | D | Giorgio Armani SpA    Num of records:  16    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  4500444636    Adv. Name:  HDM | D | Bermuda Department of Tourism    Num of records:  18    List of cat/segm:  ['i_201', 'i_214', 'i_232', 'i_653', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  4506183092    Adv. Name:  ALL | R | Google DoubleClick Ad Exchange (Programmatic Guaranteed)    Num of records:  5207    List of cat/segm:  ['cm_0012', 'cm_0037', 'cm_0063', 'cm_0070', 'cm_0080', 'cm_0084', 'cm_0092', 'cm_0096', 'cm_0101', 'cm_0106', 'cm_0110', 'cm_0112', 'cm_0116', 'cm_0120', 'cm_0126', 'cm_0127', 'cm_0137', 'cm_0146', 'cm_0149', 'cm_0152', 'cm_0157', 'cm_0167', 'cm_0186', 'cm_0191', 'cm_0201', 'i_158', 'i_159', 'i_163', 'i_177', 'i_179', 'i_186', 'i_191', 'i_192', 'i_193', 'i_1KXCLD', 'i_201', 'i_208', 'i_209', 'i_210', 'i_211', 'i_214', 'i_215', 'i_216', 'i_218', 'i_221', 'i_223', 'i_224', 'i_225', 'i_226', 'i_227', 'i_228', 'i_229', 'i_23', 'i_230', 'i_231', 'i_232', 'i_233', 'i_234', 'i_235', 'i_238', 'i_257', 'i_274', 'i_275', 'i_276', 'i_279', 'i_280', 'i_283', 'i_284', 'i_285', 'i_323', 'i_324', 'i_338', 'i_422', 'i_432', 'i_434', 'i_435', 'i_436', 'i_437', 'i_438', 'i_439', 'i_440', 'i_473', 'i_474', 'i_476', 'i_477', 'i_478', 'i_481', 'i_482', 'i_483', 'i_492', 'i_542', 'i_544', 'i_549', 'i_552', 'i_555', 'i_557', 'i_558', 'i_559', 'i_560', 'i_563', 'i_564', 'i_565', 'i_567', 'i_570', 'i_571', 'i_572', 'i_573', 'i_575', 'i_576', 'i_578', 'i_580', 'i_581', 'i_582', 'i_583', 'i_584', 'i_585', 'i_586', 'i_588', 'i_589', 'i_590', 'i_591', 'i_593', 'i_594', 'i_595', 'i_596', 'i_628', 'i_634', 'i_640', 'i_653', 'i_654', 'i_655', 'i_663', 'i_664', 'i_666', 'i_667', 'i_668', 'i_8VZQHL', 'i_97', 'i_JLBCU7', 'i_286', 'i_320', 'i_295', 'i_310', 'i_318']
# Adv ID:  4565651175    Adv. Name:  HDM | D | Ikea U S Incorporated    Num of records:  237    List of cat/segm:  ['cm_0097', 'cm_0169', 'i_257', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285']
# Adv ID:  4572943415    Adv. Name:  HDM | D | Eli Lilly & Company/Verzenio    Num of records:  1107    List of cat/segm:  ['cm_0183', 'i_123', 'i_186', 'i_223', 'i_238', 'i_279', 'i_386', 'i_566']
# Adv ID:  4598214154    Adv. Name:  HDM | D | King's Hawaiian Bakery West    Num of records:  1088    List of cat/segm:  ['cm_0178', 'cm_0179', 'i_216', 'i_279']
# Adv ID:  4615631691    Adv. Name:  JAM | D | Kia    Num of records:  541    List of cat/segm:  ['i_150', 'i_22', 'i_223', 'i_23', 'i_37', 'i_38', 'i_39', 'i_40', 'i_6', 'i_670']
# Adv ID:  4615636311    Adv. Name:  JAM | D | Mercedes - Regional    Num of records:  28    List of cat/segm:  ['i_123', 'i_125', 'i_127', 'i_128', 'i_129', 'i_130', 'i_221', 'i_223', 'i_225', 'i_227', 'i_231', 'i_234', 'i_492', 'i_544', 'i_549', 'i_567', 'i_583', 'i_596', 'i_676']
# Adv ID:  4616004467    Adv. Name:  JAM | D | Toyota    Num of records:  2595    List of cat/segm:  ['cm_0012', 'cm_0112', 'cm_0126', 'cm_0131', 'i_158', 'i_163', 'i_179', 'i_186', 'i_191', 'i_192', 'i_193', 'i_201', 'i_221', 'i_223', 'i_224', 'i_225', 'i_227', 'i_23', 'i_231', 'i_234', 'i_422', 'i_432', 'i_434', 'i_435', 'i_436', 'i_437', 'i_438', 'i_439', 'i_440', 'i_482', 'i_492', 'i_544', 'i_549', 'i_552', 'i_560', 'i_561', 'i_565', 'i_566', 'i_573', 'i_575', 'i_576', 'i_577', 'i_578', 'i_579', 'i_580', 'i_581', 'i_582', 'i_589', 'i_596', 'i_666', 'i_667', 'i_8VZQHL', 'i_JLBCU7']
# Adv ID:  4616007971    Adv. Name:  JAM | D | Volvo - Regional    Num of records:  1638    List of cat/segm:  ['cm_0043', 'i_1', 'i_132', 'i_150', 'i_177', 'i_192', 'i_223', 'i_225', 'i_243', 'i_283', 'i_391', 'i_467', 'i_483', 'i_492', 'i_52', 'i_53', 'i_596', 'i_653']
# Adv ID:  4616322139    Adv. Name:  JAM | D | Genesis    Num of records:  135    List of cat/segm:  ['i_1', 'i_159', 'i_16', 'i_160', 'i_201', 'i_22', 'i_23', 'i_25', 'i_30', 'i_512', 'i_552', 'i_557', 'i_565', 'i_576', 'i_6', 'i_653', 'i_665', 'i_677', 'i_78']
# Adv ID:  4616325016    Adv. Name:  JAM | D | Lexus    Num of records:  90    List of cat/segm:  ['cm_0126', 'i_158', 'i_163', 'i_179', 'i_201', 'i_221', 'i_223', 'i_225', 'i_227', 'i_23', 'i_231', 'i_234', 'i_492', 'i_544', 'i_549', 'i_565', 'i_576', 'i_653', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668', 'i_8VZQHL']
# Adv ID:  4616329402    Adv. Name:  JAM | R | Toyota - Programmatic    Num of records:  173    List of cat/segm:  ['cm_0013', 'cm_0073', 'cm_0077', 'i_14', 'i_15', 'i_221', 'i_223', 'i_225', 'i_226', 'i_227', 'i_228', 'i_229', 'i_230', 'i_483', 'i_492', 'i_498', 'i_499', 'i_500', 'i_502', 'i_504', 'i_505', 'i_506', 'i_511', 'i_514', 'i_526', 'i_530', 'i_531', 'i_548', 'i_549', 'i_665', 'i_677']
# Adv ID:  4616330323    Adv. Name:  JAM | D | Volvo    Num of records:  4654    List of cat/segm:  ['cm_0043', 'i_201', 'i_207']
# Adv ID:  4617238548    Adv. Name:  HMIUK | R | Programmatic Direct    Num of records:  440    List of cat/segm:  ['i_210', 'i_211', 'i_214', 'i_216', 'i_218', 'i_221']
# Adv ID:  4623955896    Adv. Name:  HDM | D | Harry Winston Incorporated    Num of records:  25    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  4633107093    Adv. Name:  ALL | R | Google DoubleClick Ad Exchange (Preferred Deal)    Num of records:  11576    List of cat/segm:  ['cm_0005', 'cm_0012', 'cm_0013', 'cm_0024', 'cm_0034', 'cm_0048', 'cm_0056', 'cm_0057', 'cm_0063', 'cm_0084', 'cm_0088', 'cm_0096', 'cm_0101', 'cm_0108', 'cm_0112', 'cm_0123', 'cm_0126', 'cm_0131', 'cm_0132', 'cm_0133', 'cm_0146', 'cm_0151', 'cm_0155', 'cm_0157', 'cm_0163', 'cm_0179', 'cm_0180', 'cm_0186', 'cm_0208', 'i_1', 'i_106', 'i_112', 'i_123', 'i_125', 'i_126', 'i_127', 'i_128', 'i_129', 'i_130', 'i_137', 'i_157', 'i_158', 'i_159', 'i_16', 'i_160', 'i_162', 'i_163', 'i_164', 'i_165', 'i_168', 'i_170', 'i_172', 'i_179', 'i_186', 'i_188', 'i_191', 'i_192', 'i_193', 'i_194', 'i_195', 'i_196', 'i_197', 'i_198', 'i_199', 'i_1KXCLD', 'i_201', 'i_203', 'i_210', 'i_211', 'i_212', 'i_213', 'i_214', 'i_215', 'i_216', 'i_217', 'i_218', 'i_219', 'i_22', 'i_220', 'i_221', 'i_222', 'i_223', 'i_224', 'i_225', 'i_226', 'i_227', 'i_228', 'i_229', 'i_23', 'i_230', 'i_231', 'i_232', 'i_234', 'i_235', 'i_238', 'i_25', 'i_252', 'i_257', 'i_269', 'i_274', 'i_275', 'i_276', 'i_278', 'i_279', 'i_280', 'i_283', 'i_284', 'i_285', 'i_30', 'i_323', 'i_324', 'i_344', 'i_399', 'i_403', 'i_422', 'i_424', 'i_425', 'i_426', 'i_427', 'i_428', 'i_429', 'i_431', 'i_432', 'i_434', 'i_435', 'i_436', 'i_437', 'i_438', 'i_439', 'i_440', 'i_441', 'i_445', 'i_451', 'i_473', 'i_476', 'i_477', 'i_478', 'i_481', 'i_482', 'i_483', 'i_484', 'i_485', 'i_486', 'i_487', 'i_488', 'i_489', 'i_490', 'i_492', 'i_493', 'i_495', 'i_499', 'i_500', 'i_502', 'i_503', 'i_504', 'i_505', 'i_506', 'i_509', 'i_510', 'i_511', 'i_512', 'i_513', 'i_514', 'i_515', 'i_516', 'i_517', 'i_518', 'i_519', 'i_520', 'i_521', 'i_522', 'i_523', 'i_525', 'i_526', 'i_530', 'i_531', 'i_533', 'i_534', 'i_535', 'i_536', 'i_537', 'i_539', 'i_540', 'i_541', 'i_542', 'i_543', 'i_544', 'i_545', 'i_546', 'i_547', 'i_548', 'i_549', 'i_550', 'i_552', 'i_553', 'i_555', 'i_556', 'i_557', 'i_558', 'i_559', 'i_560', 'i_561', 'i_565', 'i_566', 'i_567', 'i_571', 'i_572', 'i_573', 'i_575', 'i_576', 'i_577', 'i_578', 'i_579', 'i_58', 'i_580', 'i_581', 'i_582', 'i_583', 'i_589', 'i_590', 'i_591', 'i_596', 'i_640', 'i_653', 'i_654', 'i_655', 'i_663', 'i_664', 'i_665', 'i_666', 'i_667', 'i_668', 'i_676', 'i_677', 'i_685', 'i_78', 'i_8VZQHL', 'i_91', 'i_JLBCU7', 'i_286', 'i_453', 'i_320']
# Adv ID:  4652320986    Adv. Name:  HMI | R | INVIBES    Num of records:  329    List of cat/segm:  ['cm_0066']
# Adv ID:  4654668465    Adv. Name:  HDM | D | Prada SpA Group/Miu Miu Apparel    Num of records:  6    List of cat/segm:  ['cm_0008', 'i_560', 'i_576', 'i_577', 'i_578', 'i_579']
# Adv ID:  4730219032    Adv. Name:  HDM | D | Amgen Incorporated/Amgen Incorporated    Num of records:  317    List of cat/segm:  ['i_225', 'i_230', 'i_236']
# Adv ID:  4744824902    Adv. Name:  JAM | R | Mazda - Programmatic    Num of records:  641    List of cat/segm:  ['cm_0034', 'cm_0095', 'cm_0113', 'cm_0158', 'cm_0169', 'i_158', 'i_159', 'i_16', 'i_160', 'i_163', 'i_179', 'i_22', 'i_225', 'i_25', 'i_257', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285', 'i_30', 'i_557', 'i_665', 'i_677', 'i_78', 'i_8VZQHL']
# Adv ID:  4768802805    Adv. Name:  HDM | D | Farrow and Ball    Num of records:  43    List of cat/segm:  ['cm_0169', 'i_257', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285']
# Adv ID:  4802048131    Adv. Name:  HDM | D | Tumi Corporation/Luggage    Num of records:  27    List of cat/segm:  ['cm_0060', 'cm_0180', 'i_483', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  4810545812    Adv. Name:  JAM | D | BP / Castrol    Num of records:  290    List of cat/segm:  ['cm_0005', 'cm_0009', 'cm_0056', 'cm_0132', 'i_25', 'i_28', 'i_32', 'i_34', 'i_518', 'i_519', 'i_670', 'i_680']
# Adv ID:  4822998862    Adv. Name:  HDM | D | BioFilm Incorporated/Astroglide    Num of records:  42    List of cat/segm:  ['i_188', 'i_191']
# Adv ID:  4839643151    Adv. Name:  HDM | D | Bulgari Corporation of America/Bulgari Jewelry    Num of records:  233    List of cat/segm:  ['cm_0027', 'cm_0101', 'cm_0133', 'cm_0177', 'i_201', 'i_23', 'i_231', 'i_237', 'i_383', 'i_386', 'i_389', 'i_433', 'i_437', 'i_438', 'i_476', 'i_478', 'i_565', 'i_576', 'i_316', 'i_322', 'i_453', 'i_320', 'i_315']
# Adv ID:  4854434784    Adv. Name:  HDM | D | Kering/Gucci America    Num of records:  168    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  4855555657    Adv. Name:  HDM | D | Richemont Gro/Vendome Luxu/Van Cleef & A    Num of records:  910    List of cat/segm:  ['cm_0027', 'cm_0139', 'cm_0151', 'cm_0177', 'i_102', 'i_202', 'i_203', 'i_204', 'i_205', 'i_206', 'i_207', 'i_208', 'i_209', 'i_231', 'i_237', 'i_252', 'i_262', 'i_274', 'i_275', 'i_283', 'i_284', 'i_285', 'i_383', 'i_386', 'i_389', 'i_42', 'i_43', 'i_433', 'i_437', 'i_438', 'i_439', 'i_473', 'i_476', 'i_478', 'i_48', 'i_49', 'i_552', 'i_560', 'i_561', 'i_562', 'i_563', 'i_564', 'i_565', 'i_566', 'i_567', 'i_568', 'i_569', 'i_570', 'i_571', 'i_572', 'i_573', 'i_578', 'i_580', 'i_581', 'i_653', 'i_654', 'i_655', 'i_656', 'i_657', 'i_658', 'i_659', 'i_660', 'i_661', 'i_662', 'i_663', 'i_664', 'i_665', 'i_666', 'i_667', 'i_316', 'i_322', 'i_453', 'i_320', 'i_315']
# Adv ID:  4861913238    Adv. Name:  HDM | PG | Google DoubleClick Ad Exchange (Programmatic Guaranteed) (Globe & Mail - Canada Geo Sales)    Num of records:  461    List of cat/segm:  ['cm_0048', 'cm_0101', 'cm_0137', 'i_1', 'i_109', 'i_16', 'i_190', 'i_221', 'i_223', 'i_224', 'i_225', 'i_226', 'i_227', 'i_228', 'i_229', 'i_230', 'i_231', 'i_232', 'i_233', 'i_234', 'i_235', 'i_236', 'i_237', 'i_238', 'i_324', 'i_338', 'i_386', 'i_388', 'i_389', 'i_433', 'i_483', 'i_596', 'i_640', 'i_653', 'i_666', 'i_667', 'i_668', 'i_670', 'i_671', 'i_672', 'i_673', 'i_674', 'i_676', 'i_677', 'i_678', 'i_97', 'i_JLBCU7', 'i_320']
# Adv ID:  4862094632    Adv. Name:  HDM | PD | Google DoubleClick Ad Exchange (Preferred Deals) (Globe & Mail - Canada Geo Sales)    Num of records:  316    List of cat/segm:  ['i_159', 'i_186', 'i_192', 'i_197', 'i_198', 'i_210', 'i_211', 'i_275', 'i_279', 'i_285', 'i_680', 'i_682', 'i_683', 'i_684', 'i_95']
# Adv ID:  4865869187    Adv. Name:  JAM | R | Volvo - Programmatic    Num of records:  10    List of cat/segm:  ['cm_0078', 'cm_0095', 'cm_0129', 'i_159', 'i_160', 'i_22', 'i_557', 'i_596', 'i_665', 'i_677', 'i_78']
# Adv ID:  4889157723    Adv. Name:  HDM | D | Athletic Brewing Company    Num of records:  76    List of cat/segm:  ['i_222', 'i_225', 'i_229', 'i_483']
# Adv ID:  4889856958    Adv. Name:  HDM | D | Liberty Ventures Gro/Starz Entertainment    Num of records:  11    List of cat/segm:  ['cm_0036', 'i_324', 'i_640', 'i_JLBCU7']
# Adv ID:  4914074582    Adv. Name:  HDM | D | Tillamook County Creamery Assn/Tillamook Dairy Products    Num of records:  71    List of cat/segm:  ['cm_0032', 'cm_0084', 'cm_0086', 'cm_0116', 'cm_0126', 'cm_0146', 'cm_0170', 'cm_0186', 'i_161', 'i_163', 'i_1KXCLD', 'i_210', 'i_211', 'i_214', 'i_215', 'i_216', 'i_217', 'i_218', 'i_221', 'i_223', 'i_225', 'i_227', 'i_231', 'i_234', 'i_279', 'i_283', 'i_478', 'i_483', 'i_492', 'i_544', 'i_549', 'i_653']
# Adv ID:  4914198657    Adv. Name:  HDM | D | Ulta    Num of records:  227    List of cat/segm:  ['cm_0088', 'i_555', 'i_556', 'i_557', 'i_558']
# Adv ID:  4918213170    Adv. Name:  HDM | D | Tapestry Incorporated/Coach Incorporated    Num of records:  307    List of cat/segm:  ['cm_0008', 'cm_0141', 'i_162', 'i_165', 'i_186', 'i_188', 'i_432', 'i_560', 'i_576', 'i_577', 'i_578', 'i_579', 'i_93', 'i_JLBCU7']
# Adv ID:  4919712629    Adv. Name:  HDM | D | Burberrys International/Burberry Accessories    Num of records:  19    List of cat/segm:  ['cm_0177', 'i_476', 'i_478']
# Adv ID:  4926028027    Adv. Name:  HDM | D | Richemont Gro/Vendome Luxu/Montblanc Nor/Montblanc (EXCH)    Num of records:  163    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  4926162929    Adv. Name:  HDM | D | Henri Stern Watch/Patek Philippe    Num of records:  35    List of cat/segm:  ['i_115', 'i_23', 'i_483', 'i_576', 'i_579', 'i_580', 'i_581', 'i_583', 'i_596', 'i_653', 'i_680']
# Adv ID:  4935158320    Adv. Name:  HDM | D | Boucheron Joaillier    Num of records:  29    List of cat/segm:  ['cm_0027', 'cm_0133', 'i_201', 'i_23', 'i_231', 'i_237', 'i_383', 'i_386', 'i_389', 'i_433', 'i_437', 'i_438', 'i_565', 'i_576', 'i_316', 'i_322', 'i_453', 'i_320', 'i_315']
# Adv ID:  4943611888    Adv. Name:  HDM | D | Comcast Entertainmen/NBC Universal Media/Peacock TV LLC    Num of records:  127    List of cat/segm:  ['i_324', 'i_640', 'i_JLBCU7']
# Adv ID:  4944552621    Adv. Name:  HDM | D | Kering/Pomellato USA    Num of records:  55    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  4951828544    Adv. Name:  HDM | D | Richemont Gro/Vendome Luxu/Vacheron Cons    Num of records:  14    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  4958451038    Adv. Name:  HDM | D | The Swatch Group SA/SMH-US Incorporated/Longines Watches    Num of records:  115    List of cat/segm:  ['cm_0008', 'cm_0113', 'cm_0180', 'i_16', 'i_25', 'i_30', 'i_560', 'i_576', 'i_577', 'i_578', 'i_579', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  4959011004    Adv. Name:  HDM | R | Hearst Magazines Brand Licensing Paid Programs    Num of records:  119    List of cat/segm:  ['cm_0063', 'cm_0177', 'i_422', 'i_423', 'i_424', 'i_425', 'i_426', 'i_427', 'i_428', 'i_429', 'i_430', 'i_431', 'i_476', 'i_478']
# Adv ID:  4961855034    Adv. Name:  HDM | D | UBS Group AG/UBS    Num of records:  464    List of cat/segm:  ['cm_0119', 'i_125', 'i_126', 'i_127', 'i_128', 'i_129', 'i_130', 'i_386', 'i_387', 'i_388', 'i_567', 'i_583', 'i_676']
# Adv ID:  4981466347    Adv. Name:  HDM | D | Formica Corporation    Num of records:  37    List of cat/segm:  ['cm_0084', 'cm_0086', 'cm_0116', 'cm_0186', 'i_112', 'i_161', 'i_163', 'i_1KXCLD', 'i_210', 'i_211', 'i_215', 'i_216', 'i_217', 'i_257', 'i_274', 'i_275', 'i_276', 'i_279', 'i_280', 'i_283', 'i_284', 'i_285', 'i_445', 'i_451']
# Adv ID:  4996224902    Adv. Name:  HDM | D | Novartis Pharma AG/Cosentyx    Num of records:  282    List of cat/segm:  ['i_211', 'i_386', 'i_521', 'i_522', 'i_523']
# Adv ID:  5002046810    Adv. Name:  HDM | D | Logitech/Logitech Harmony    Num of records:  12    List of cat/segm:  ['i_223', 'i_224', 'i_225', 'i_226', 'i_227', 'i_228', 'i_229', 'i_230', 'i_231', 'i_232', 'i_233', 'i_234', 'i_235', 'i_236', 'i_237', 'i_238', 'i_554', 'i_555', 'i_556', 'i_557', 'i_558']
# Adv ID:  5026426302    Adv. Name:  HDM | D | LVMH Moet Hennessy Lou Vuitto/Loro Piana    Num of records:  82    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  5027701611    Adv. Name:  HDM | D | Graff Diamonds    Num of records:  12    List of cat/segm:  ['cm_0008', 'cm_0133', 'i_201', 'i_23', 'i_560', 'i_565', 'i_576', 'i_577', 'i_578', 'i_579']
# Adv ID:  5029821539    Adv. Name:  HDM | D | Neurocrine Biosciences Inc/Neurocrine Biosciences Inc    Num of records:  369    List of cat/segm:  ['cm_0096', 'cm_0101', 'cm_0153', 'i_223', 'i_224', 'i_228', 'i_229', 'i_230', 'i_235', 'i_238', 'i_286', 'i_287', 'i_320']
# Adv ID:  5032064810    Adv. Name:  HDM | D | RPM Incorporated/Rust-Oleum Corporation    Num of records:  33    List of cat/segm:  ['i_257', 'i_274', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285']
# Adv ID:  5033308496    Adv. Name:  HDM | D | Tacori    Num of records:  14    List of cat/segm:  ['cm_0177', 'i_476', 'i_478']
# Adv ID:  5034218359    Adv. Name:  HDM | D | Capri Holdings/Michael Kors USA Incorpor    Num of records:  51    List of cat/segm:  ['cm_0008', 'cm_0177', 'i_476', 'i_478', 'i_560', 'i_576', 'i_577', 'i_578', 'i_579']
# Adv ID:  5049318629    Adv. Name:  HDM | D | LVMH Moet Hennessy Lou/Tiffany & Company    Num of records:  104    List of cat/segm:  ['cm_0027', 'cm_0133', 'i_201', 'i_23', 'i_231', 'i_237', 'i_383', 'i_386', 'i_389', 'i_433', 'i_437', 'i_438', 'i_565', 'i_576', 'i_316', 'i_322', 'i_453', 'i_320', 'i_315']
# Adv ID:  5068181669    Adv. Name:  HMIUK | D | Apple (UK) Ltd    Num of records:  258    List of cat/segm:  ['cm_0058', 'cm_0084', 'i_1KXCLD', 'i_210']
# Adv ID:  5087961319    Adv. Name:  HDM | D | Texas Economic Develop Tourism/Texas Tourism    Num of records:  468    List of cat/segm:  ['cm_0180', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  5091231330    Adv. Name:  HDM | D | Laurice & Company/Bond No 9 Fragrances    Num of records:  201    List of cat/segm:  ['cm_0041', 'cm_0180', 'i_558', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  5093053339    Adv. Name:  HMIUK | D | Bvlgari (UK) Ltd    Num of records:  14    List of cat/segm:  ['cm_0141', 'i_162']
# Adv ID:  5175166928    Adv. Name:  HDM | D | Sanderson Design Gro/Walker Greenbank PL/Zoffany Limited    Num of records:  42    List of cat/segm:  ['i_112', 'i_276', 'i_280', 'i_283', 'i_445', 'i_451']
# Adv ID:  5178614553    Adv. Name:  HDM | D | NBC/Hulu/Hulu    Num of records:  31    List of cat/segm:  ['i_1KXCLD']
# Adv ID:  5189247492    Adv. Name:  HDM | D | Endo Pharmaceuticals Incorporated/Xiaflex    Num of records:  197    List of cat/segm:  ['cm_0096', 'i_228', 'i_464', 'i_JLBCU7', 'i_286']
# Adv ID:  5194422466    Adv. Name:  HDM | D | Ben Bridge Jewelers/Ben Bridge    Num of records:  60    List of cat/segm:  ['cm_0133', 'cm_0177', 'cm_0180', 'i_201', 'i_23', 'i_476', 'i_478', 'i_565', 'i_576', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  5199034039    Adv. Name:  HDM | D | Target Corporation/Shipt Incorporated    Num of records:  221    List of cat/segm:  ['cm_0084', 'cm_0086', 'cm_0116', 'cm_0126', 'cm_0146', 'cm_0186', 'i_161', 'i_163', 'i_1KXCLD', 'i_210', 'i_216', 'i_278', 'i_279', 'i_283', 'i_477', 'i_478', 'i_653']
# Adv ID:  5200391906    Adv. Name:  HDM | D | US Government Recruiting/Army Reserve    Num of records:  65    List of cat/segm:  ['i_52', 'i_628', 'i_77']
# Adv ID:  5221632410    Adv. Name:  HDM | D | Johnson Health Tech/Horizon Fitness    Num of records:  37    List of cat/segm:  ['i_221', 'i_223', 'i_225', 'i_227', 'i_231', 'i_234', 'i_492', 'i_544', 'i_549']
# Adv ID:  5238094921    Adv. Name:  HDM | D | Nonancourt Family Trust/Laurent Perrier    Num of records:  72    List of cat/segm:  ['i_210', 'i_211', 'i_473', 'i_552', 'i_553', 'i_559', 'i_596', 'i_653']
# Adv ID:  5279212991    Adv. Name:  HDM | D | EOS Hospitality/L'Ermitage Beverly Hills    Num of records:  56    List of cat/segm:  ['cm_0176', 'i_655', 'i_671']
# Adv ID:  5286918206    Adv. Name:  HDM | R | KO_LGA Media Direct IO Deals    Num of records:  392    List of cat/segm:  ['i_552', 'i_553', 'i_560', 'i_566', 'i_579', 'i_582', 'i_595']
# Adv ID:  5338051613    Adv. Name:  HDM | D | Marc Fisher Footwear/Easy Spirit    Num of records:  8    List of cat/segm:  ['i_227', 'i_232', 'i_238', 'i_492', 'i_542', 'i_571', 'i_572']
# Adv ID:  5338841947    Adv. Name:  HDM | D | Ragnar Events LLC/Ragnar    Num of records:  9    List of cat/segm:  ['i_159', 'i_160', 'i_22', 'i_221', 'i_223', 'i_225', 'i_227', 'i_231', 'i_234', 'i_492', 'i_544', 'i_549', 'i_557', 'i_665', 'i_677', 'i_78']
# Adv ID:  5342095356    Adv. Name:  HDM | D | Swedish Match North America LLC/ZYN Nicotine Pouches    Num of records:  2732    List of cat/segm:  ['cm_0012', 'cm_0096', 'i_186', 'i_198', 'i_223', 'i_483', 'i_286']
# Adv ID:  5343117008    Adv. Name:  HDM | D | Humana Inc/CenterWell    Num of records:  5    List of cat/segm:  ['i_109', 'i_190', 'i_230', 'i_416']
# Adv ID:  5346376308    Adv. Name:  HDM | D | Quaker Oats Company/Near East    Num of records:  26    List of cat/segm:  ['i_210', 'i_212', 'i_213', 'i_216', 'i_219', 'i_220', 'i_221', 'i_229', 'i_231', 'i_278', 'i_477']
# Adv ID:  5351164937    Adv. Name:  HDM | D | Ipsen/Ipsen    Num of records:  51    List of cat/segm:  ['cm_0096', 'cm_0101', 'i_223', 'i_230', 'i_286', 'i_320']
# Adv ID:  5353222062    Adv. Name:  HDM | D | Incyte Corporation/Opzelura    Num of records:  29    List of cat/segm:  ['cm_0012', 'cm_0037', 'cm_0096', 'cm_0101', 'cm_0112', 'cm_0157', 'i_186', 'i_191', 'i_192', 'i_193', 'i_223', 'i_224', 'i_228', 'i_229', 'i_230', 'i_232', 'i_235', 'i_238', 'i_323', 'i_422', 'i_482', 'i_557', 'i_559', 'i_575', 'i_590', 'i_591', 'i_666', 'i_667', 'i_286', 'i_320']
# Adv ID:  5358255291    Adv. Name:  HDM | D | Hermes/Hermes Paris    Num of records:  37    List of cat/segm:  ['i_201', 'i_232', 'i_432', 'i_476', 'i_552', 'i_565', 'i_576']
# Adv ID:  5361972697    Adv. Name:  HDM | D | Daniel Swarovski/Swarovski Consumer Goods Bus    Num of records:  59    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  5363762520    Adv. Name:  HDM | D | Pfizer Incorporated/Pfizer Incorporated    Num of records:  131    List of cat/segm:  ['i_192']
# Adv ID:  5365052570    Adv. Name:  HDM | D | MGM Resorts International/The Cosmopolitan of Las Vegas    Num of records:  9    List of cat/segm:  ['i_181', 'i_JLBCU7']
# Adv ID:  5403699739    Adv. Name:  HDM | D | Sanofi/Icy Hot    Num of records:  298    List of cat/segm:  ['i_236', 'i_483']
# Adv ID:  5408555435    Adv. Name:  HDM | D | Pooky Lighting Ltd/Pooky Lighting Ltd    Num of records:  32    List of cat/segm:  ['i_112', 'i_276', 'i_280', 'i_283', 'i_445', 'i_451']
# Adv ID:  5411245192    Adv. Name:  HDM | D | The Hartford/The Hartford Insurance Company    Num of records:  73    List of cat/segm:  ['i_123', 'i_125', 'i_126', 'i_127', 'i_128', 'i_129', 'i_130', 'i_221', 'i_223', 'i_225', 'i_227', 'i_231', 'i_234', 'i_492', 'i_544', 'i_549', 'i_567', 'i_583', 'i_676']
# Adv ID:  5427441825    Adv. Name:  HDM | D | Only1 Brands Incorporated    Num of records:  428    List of cat/segm:  ['cm_0088', 'i_553', 'i_555', 'i_556', 'i_557', 'i_558']
# Adv ID:  5428260470    Adv. Name:  HDM | D | Bogle Family Vineyards/Bogle Family Vineyards    Num of records:  100    List of cat/segm:  ['i_210', 'i_211', 'i_552', 'i_560', 'i_561', 'i_566', 'i_573', 'i_576', 'i_577', 'i_578', 'i_579', 'i_580', 'i_582', 'i_589']
# Adv ID:  5434489431    Adv. Name:  HDM | D | Bausch & Lomb Incorporated/Preservision    Num of records:  280    List of cat/segm:  ['cm_0096', 'cm_0101', 'i_216', 'i_223', 'i_224', 'i_228', 'i_229', 'i_230', 'i_235', 'i_238', 'i_653', 'i_JLBCU7', 'i_286', 'i_320']
# Adv ID:  5445701743    Adv. Name:  HDM | D | Overstock.com    Num of records:  31    List of cat/segm:  ['i_112', 'i_276', 'i_280', 'i_283', 'i_445', 'i_451']
# Adv ID:  5445722041    Adv. Name:  HDM | D | CellResearch Corporation Pte Ltd/Calecim Professional    Num of records:  39    List of cat/segm:  ['cm_0153', 'i_223', 'i_287']
# Adv ID:  5482313747    Adv. Name:  HDM | D | Habitat for Humanity International    Num of records:  73    List of cat/segm:  ['i_399', 'i_8YPBBL']
# Adv ID:  5501675213    Adv. Name:  HDM | D | TRT Holdings Incorporated/The Omni Homestead Resort    Num of records:  86    List of cat/segm:  ['cm_0012', 'cm_0112', 'i_186', 'i_191', 'i_192', 'i_193', 'i_201', 'i_224', 'i_23', 'i_422', 'i_482', 'i_565', 'i_575', 'i_576', 'i_666', 'i_667']
# Adv ID:  5506329810    Adv. Name:  HDM | D | Eisai Company Limited/Eisai Company Limited    Num of records:  578    List of cat/segm:  ['i_190']
# Adv ID:  5564951004    Adv. Name:  HDM | D | A.O. Smith Corporation    Num of records:  63    List of cat/segm:  ['i_112', 'i_159', 'i_160', 'i_22', 'i_257', 'i_274', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285', 'i_445', 'i_451', 'i_557', 'i_665', 'i_677', 'i_78']
# Adv ID:  5613960123    Adv. Name:  HDM | D | Victorinox Swiss Ar/Victorinox Swiss Arm    Num of records:  599    List of cat/segm:  ['cm_0106', 'i_201', 'i_23', 'i_257', 'i_274', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285', 'i_476', 'i_478', 'i_552', 'i_557', 'i_560', 'i_561', 'i_565', 'i_566', 'i_573', 'i_576', 'i_577', 'i_578', 'i_579', 'i_580', 'i_582', 'i_589', 'i_653', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  5616684888    Adv. Name:  HDM | D | Puerto Rico Tourism/Discover Puerto Rico    Num of records:  123    List of cat/segm:  ['i_159', 'i_160', 'i_22', 'i_557', 'i_653', 'i_654', 'i_655', 'i_663', 'i_664', 'i_665', 'i_668', 'i_677', 'i_78']
# Adv ID:  5616812825    Adv. Name:  HDM | D | Lundbeck LLC/Vyepti    Num of records:  1513    List of cat/segm:  ['i_165', 'i_196', 'i_197', 'i_451', 'i_653']
# Adv ID:  5617360557    Adv. Name:  HDM | D | DoorDash Incorporated/Doordash Incorporated    Num of records:  887    List of cat/segm:  ['cm_0002', 'cm_0024', 'cm_0178', 'i_1KXCLD', 'i_279']
# Adv ID:  5621237456    Adv. Name:  HDM | D | Colgate Personal Care Pro/Colgate Brands/Total Toothpaste    Num of records:  203    List of cat/segm:  ['cm_0096', 'cm_0101', 'i_223', 'i_224', 'i_228', 'i_229', 'i_230', 'i_235', 'i_238', 'i_286', 'i_320']
# Adv ID:  5622862021    Adv. Name:  HDM | D | LVMH Moet Hennessy Lou Vuitton    Num of records:  96    List of cat/segm:  ['cm_0133', 'cm_0161', 'i_201', 'i_23', 'i_231', 'i_237', 'i_383', 'i_386', 'i_389', 'i_565', 'i_576', 'i_316', 'i_322', 'i_453', 'i_320', 'i_315']
# Adv ID:  5627481513    Adv. Name:  HDM | D | Vertex Pharmaceutical Inc/Vertex Pharmaceutical Inc    Num of records:  163    List of cat/segm:  ['cm_0096', 'i_286']
# Adv ID:  5629850861    Adv. Name:  HDM | D | Perfumania Ho/Parlux Fragra/Kenneth Cole    Num of records:  99    List of cat/segm:  ['i_476', 'i_558']
# Adv ID:  5630884701    Adv. Name:  HMIES | D | YSABEL MORA    Num of records:  6    List of cat/segm:  ['i_552', 'i_559', 'i_560']
# Adv ID:  5632533831    Adv. Name:  HDM | D | Intercontinental Hotels Group/Intercontinental Htls & Resort    Num of records:  10    List of cat/segm:  ['i_201', 'i_23', 'i_565', 'i_576', 'i_653', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  5632534293    Adv. Name:  HDM | D | Visit Dallas/Visit Dallas    Num of records:  49    List of cat/segm:  ['i_179', 'i_210', 'i_214', 'i_216', 'i_218', 'i_279', 'i_653', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  5633161867    Adv. Name:  HDM | D | UCB Pharma Incorporated/Bimzelx    Num of records:  134    List of cat/segm:  ['i_109', 'i_223', 'i_238']
# Adv ID:  5638441324    Adv. Name:  HDM | PD | Boston Beer (SFID=0014X00002KBZnFQAX)    Num of records:  25    List of cat/segm:  ['cm_0172', 'i_211']
# Adv ID:  5639321919    Adv. Name:  HDM | D | Lumber Liquidators    Num of records:  32    List of cat/segm:  ['cm_0169', 'i_257', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285']
# Adv ID:  5641584647    Adv. Name:  HDM | D | Brookstone/Brookstone    Num of records:  20    List of cat/segm:  ['cm_0153', 'i_223', 'i_287']
# Adv ID:  5641811039    Adv. Name:  HDM | D | James Hardie Building Products/Siding Products    Num of records:  33    List of cat/segm:  ['cm_0023', 'cm_0095', 'cm_0169', 'i_112', 'i_159', 'i_160', 'i_22', 'i_257', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285', 'i_441', 'i_445', 'i_451', 'i_557', 'i_665', 'i_677', 'i_78']
# Adv ID:  5642258076    Adv. Name:  HDM | PG | JP Morgan Chase (SFID=0015000000jGlbgAAC)    Num of records:  123    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  5642549913    Adv. Name:  HDM | PG | Applebee's (SFID=0015000000tkonlAAA)    Num of records:  517    List of cat/segm:  ['cm_0081', 'cm_0118', 'cm_0163', 'cm_0178', 'cm_0179', 'cm_0184', 'i_157', 'i_179', 'i_1KXCLD', 'i_210', 'i_214', 'i_216', 'i_218', 'i_279', 'i_473', 'i_476', 'i_478', 'i_481', 'i_59', 'i_653']
# Adv ID:  5644249117    Adv. Name:  HDM | PD | Saks (SFID=0015000000PvfNRAAZ)    Num of records:  32    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  5646921779    Adv. Name:  HDM | PG | Amica Insurance (SFID=0014X00002Q9gZqQAJ)    Num of records:  442    List of cat/segm:  ['cm_0153', 'i_165', 'i_166', 'i_167', 'i_168', 'i_169', 'i_223', 'i_283', 'i_285', 'i_398', 'i_401', 'i_407', 'i_416', 'i_441', 'i_287']
# Adv ID:  5647253467    Adv. Name:  HDM | D | Luxottica Group SpA/Luxottica/Luxottica    Num of records:  100    List of cat/segm:  ['cm_0008', 'cm_0151', 'i_560', 'i_576', 'i_577', 'i_578', 'i_579']
# Adv ID:  5647995251    Adv. Name:  HDM | PG | Lowe's Companies Incorporated (SFID=0015000000KWdNDAA1)    Num of records:  226    List of cat/segm:  ['cm_0169', 'i_257', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285']
# Adv ID:  5648747505    Adv. Name:  HDM | D | AmorePacific Corporation/Sulwhasoo    Num of records:  55    List of cat/segm:  ['cm_0079', 'i_559']
# Adv ID:  5648804764    Adv. Name:  HDM | D | Thrift Books Global, LLC    Num of records:  80    List of cat/segm:  ['cm_0112', 'i_132', 'i_192', 'i_42', 'i_482', 'i_92', 'i_JLBCU7']
# Adv ID:  5648828579    Adv. Name:  HDM | PD |  Molekule (SFID=0010z00001SGH1CAAX)    Num of records:  21    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  5648829104    Adv. Name:  HDM | PD | Duke's Mayo (SFID = 00138000019eNVnAAM)    Num of records:  54    List of cat/segm:  ['cm_0034', 'cm_0178', 'cm_0179', 'i_158', 'i_163', 'i_179', 'i_216', 'i_279', 'i_8VZQHL']
# Adv ID:  5649056588    Adv. Name:  HDM | D | Bausch & Lomb Incorporated/Xiidra    Num of records:  525    List of cat/segm:  ['i_230', 'i_288', 'i_390']
# Advertiser ID 5649809955 not found in advertiser_df.
# Adv ID:  5650075827    Adv. Name:  HDM | PD | Common Spirit Health (SFID = XXXX)    Num of records:  37    List of cat/segm:  ['cm_0025', 'i_212', 'i_213', 'i_216', 'i_219', 'i_220', 'i_221', 'i_229', 'i_231']
# Adv ID:  5650374385    Adv. Name:  HDM | D | Cornerstone Building Brands/Simonton Windows and Doors    Num of records:  4    List of cat/segm:  ['cm_0023', 'cm_0169', 'i_112', 'i_257', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285', 'i_441', 'i_445', 'i_451']
# Adv ID:  5650425549    Adv. Name:  HDM | PG | LEGO Company/LEGO Systems (SFID=0010z00001WNhzfAAD)    Num of records:  385    List of cat/segm:  ['cm_0002', 'i_192', 'i_196', 'i_197', 'i_198', 'i_1KXCLD', 'i_243', 'i_338', 'i_378', 'i_473', 'i_476', 'i_478', 'i_482', 'i_666']
# Adv ID:  5654464048    Adv. Name:  HDM | PD | JetBlue (SFID = 0015000000wHd2iAAC)    Num of records:  21    List of cat/segm:  ['cm_0180', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  5655827002    Adv. Name:  HDM | D | Euroitalia USA/Versace Profumi SpA/Versace Profumo    Num of records:  122    List of cat/segm:  ['cm_0133', 'i_201', 'i_23', 'i_565', 'i_576']
# Adv ID:  5656716938    Adv. Name:  HDM | PD | Tourism Australia (SFID = 0014X00002OaeoKQAR)    Num of records:  1    List of cat/segm:  ['cm_0180', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  5656725834    Adv. Name:  HDM | PG | Samsung (SFID = 00150000010Zg9QAAS)    Num of records:  122    List of cat/segm:  ['cm_0023', 'cm_0169', 'i_112', 'i_257', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285', 'i_441', 'i_445', 'i_451']
# Adv ID:  5657648086    Adv. Name:  HDM | PG | Brown-Forman (SFID=0010z00001WMOdoAAH)    Num of records:  94    List of cat/segm:  ['cm_0172', 'i_211']
# Adv ID:  5657689375    Adv. Name:  HDM | PD | Heineken (SFID=0015000000QSlZ4AAL)    Num of records:  24    List of cat/segm:  ['cm_0158', 'i_225']
# Adv ID:  5658652980    Adv. Name:  HDM | PG | Cointreau (SFID=00150000010YbwKAAS)    Num of records:  13    List of cat/segm:  ['cm_0034', 'cm_0081', 'i_158', 'i_163', 'i_179', 'i_214', 'i_216', 'i_218', 'i_279', 'i_653', 'i_8VZQHL']
# Adv ID:  5658678916    Adv. Name:  HDM | D | Pfizer Incorporated/Velsipity    Num of records:  187    List of cat/segm:  ['i_274', 'i_432', 'i_483', 'i_484', 'i_488', 'i_508', 'i_553', 'i_JLBCU7']
# Adv ID:  5660446443    Adv. Name:  HDM | PG | Paramount (SFID=0015000000pn3p6AAA)    Num of records:  57    List of cat/segm:  ['cm_0013', 'cm_0038', 'cm_0068', 'cm_0103', 'cm_0141', 'cm_0175', 'i_162', 'i_177', 'i_209', 'i_634', 'i_93', 'i_JLBCU7', 'i_371', 'i_370']
# Adv ID:  5660979101    Adv. Name:  HDM | PD | Etsy (SFID = 0010z00001XoZPsAAN)    Num of records:  45    List of cat/segm:  ['cm_0118', 'cm_0177', 'i_473', 'i_476', 'i_478']
# Adv ID:  5661159118    Adv. Name:  HDM | PD | Volvo (SFID=00138000016sBOuAAM)    Num of records:  396    List of cat/segm:  ['cm_0023', 'cm_0057', 'cm_0060', 'cm_0095', 'cm_0113', 'cm_0133', 'cm_0164', 'i_112', 'i_159', 'i_16', 'i_160', 'i_192', 'i_201', 'i_22', 'i_23', 'i_25', 'i_276', 'i_280', 'i_283', 'i_30', 'i_441', 'i_445', 'i_451', 'i_482', 'i_483', 'i_557', 'i_565', 'i_576', 'i_665', 'i_666', 'i_677', 'i_78', 'i_JLBCU7']
# Adv ID:  5662258223    Adv. Name:  HDM | PD | Sephora (SFID=0015000000KobZIAAZ)    Num of records:  385    List of cat/segm:  ['cm_0002', 'cm_0177', 'i_1KXCLD', 'i_476', 'i_478']
# Adv ID:  5663169757    Adv. Name:  HDM | PG | Adidas (SFID = 0015000000pmdVfAAI)    Num of records:  109    List of cat/segm:  ['cm_0060', 'cm_0061', 'i_225', 'i_227', 'i_232', 'i_238', 'i_483', 'i_492', 'i_542', 'i_571', 'i_572']
# Adv ID:  5665652396    Adv. Name:  HDM | PG | ISDIN (SFID=0010z00001UYejMAAT)    Num of records:  68    List of cat/segm:  ['i_473', 'i_475', 'i_476', 'i_477', 'i_478', 'i_553', 'i_554', 'i_555', 'i_556', 'i_557', 'i_558', 'i_559']
# Adv ID:  5665768308    Adv. Name:  HDM | PD | Williams Sonoma (SFID=0015000000mHronAAC)    Num of records:  16    List of cat/segm:  ['cm_0023', 'cm_0133', 'cm_0169', 'i_112', 'i_201', 'i_23', 'i_257', 'i_275', 'i_276', 'i_280', 'i_283', 'i_284', 'i_285', 'i_441', 'i_445', 'i_451', 'i_565', 'i_576']
# Adv ID:  5670460246    Adv. Name:  HDM | PD | Mark and Graham (SFID=0014X00002O7EMKQA3)    Num of records:  24    List of cat/segm:  ['cm_0040', 'cm_0177', 'i_422', 'i_476', 'i_478']
# Adv ID:  5672540816    Adv. Name:  HDM | PG | Brown Forman Diplomatico (SFID=0014X00002KzZ5IQAV)    Num of records:  132    List of cat/segm:  ['cm_0058', 'cm_0185', 'cm_0188', 'cm_0190', 'cm_0192', 'cm_0197', 'cm_0201', 'cm_0204', 'cm_0205', 'cm_0206', 'i_1KXCLD', 'i_210', 'i_211', 'i_217']
# Adv ID:  5678078525    Adv. Name:  HDM | PG | Celebrity Cruises (SFID= 0015000000s3LkpAAE)    Num of records:  143    List of cat/segm:  ['cm_0180', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  5684059114    Adv. Name:  HDM | D | Blue Diamond Growers/Almonds    Num of records:  192    List of cat/segm:  ['cm_0002', 'i_1KXCLD']
# Adv ID:  5685076749    Adv. Name:  HDM | PD | Carnival Corporation & PLC/Carnival Cruise Lines (SFID=XXXX)    Num of records:  28    List of cat/segm:  ['cm_0180', 'i_654', 'i_655', 'i_663', 'i_664', 'i_668']
# Adv ID:  5689019211    Adv. Name:  HDM | D | Etro SpA    Num of records:  91    List of cat/segm:  ['i_552', 'i_560', 'i_561', 'i_562', 'i_563', 'i_564', 'i_565', 'i_566', 'i_577', 'i_578', 'i_579', 'i_580', 'i_581', 'i_582', 'i_583', 'i_584', 'i_585']
# Adv ID:  5689847435    Adv. Name:  HDM | D | Alphabet Incorporated/Google    Num of records:  827    List of cat/segm:  ['cm_0118', 'cm_0177', 'i_473', 'i_476', 'i_478']
# Adv ID:  5693951960    Adv. Name:  HDM | PG | L'Oreal YSL (SFID=XXXX)    Num of records:  144    List of cat/segm:  ['cm_0177', 'i_476', 'i_478']
# Adv ID:  5698857551    Adv. Name:  HDM | D | Blue Apron    Num of records:  141    List of cat/segm:  ['cm_0186', 'i_1KXCLD', 'i_220', 'i_221', 'i_232', 'i_235', 'i_477']
# Adv ID:  5698964831    Adv. Name:  HDM | PD | Wolverine Worldwide (SFID = XXXX)    Num of records:  5    List of cat/segm:  ['cm_0061', 'cm_0177', 'i_225', 'i_227', 'i_232', 'i_238', 'i_476', 'i_478', 'i_492', 'i_542', 'i_571', 'i_572']