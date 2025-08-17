import os
import sys

import pandas as pd
from numpy import int64, float32, float64, nan, where
from copy import deepcopy
from scipy.stats import chi2_contingency

def count_decimals(x):
	if pd.isnull(x):
		return nan

	s = f"{x:.20f}".rstrip('0')
	if("." in s):
		return len(s.split(".")[1])
	return 0

class Wrangler:

	def __init__(self):

		self.processType = "train"

	def read_data(self):
		if(self.processType == "train"):
			self.data = pd.read_csv(os.path.join("..", "data", "train.csv"))
		else:
			self.data = pd.read_csv(os.path.join("..", "data", "test.csv"))

	def convert_zero_visibility_to_na(self):

		self.data["Item_Visibility"] = self.data["Item_Visibility"].replace(0,nan).astype(float64)

	def count_na(self):

		nanCounts = self.data.isna().sum()
		print("\nNAN values in raw data", nanCounts)

	def fill_mappable_na(self):

		outletConflicts = (self.data.groupby("Outlet_Identifier")["Outlet_Size"].nunique(dropna=True).loc[lambda s: s > 1])
		print("Conflicting Outlet identifier & Size :", outletConflicts)

		if(outletConflicts.empty):
			print("Since no conflicts found for outlet size, proceeding with direct mapping to fill na")
			lookup = (self.data.dropna(subset=['Outlet_Size']).drop_duplicates('Outlet_Identifier', keep='first').set_index('Outlet_Identifier')['Outlet_Size'])
			self.data["Outlet_Size"] = self.data["Outlet_Size"].fillna(self.data["Outlet_Identifier"].map(lookup)).astype(object)
		else:
			print("Since conflicts were found for outlet size, proceeding with hiighest frequency value to fill na")
			grp = data.groupby("Outlet_Identifier")["Outlet_Size"]
			filler = grp.transform(lambda s: s.mode().iloc[0] if s.notna().any() else nan)
			self.data["Outlet_Size"] = self.data["Outlet_Size"].fillna(filler).astype(object)

		itemConflicts = (self.data.groupby("Item_Identifier")["Item_Weight"].nunique(dropna=True).loc[lambda s: s > 1])
		print("Conflicting Item identifier & weighte :", itemConflicts)
		if(itemConflicts.empty):
			print("Since no conflicts found for item weight, proceeding with direct mapping to fill na")
			lookup = (self.data.dropna(subset=['Item_Weight']).drop_duplicates('Item_Identifier', keep='first').set_index('Item_Identifier')['Item_Weight'])
			self.data["Item_Weight"] = self.data["Item_Weight"].fillna(self.data["Item_Identifier"].map(lookup)).astype(float64)
		else:
			print("Since conflicts were found for item weight, proceeding with hiighest frequency value to fill na")
			grp = self.data.groupby("Item_Identifier")["Item_Weight"]
			filler = grp.transform(lambda s: s.mode().iloc[0] if s.notna().any() else nan)
			self.data["Item_Weight"] = self.data["Item_Weight"].fillna(filler).astype(float64)

	def impute_case_a(self, target, predictors):
		df = self.data

		# Work on object/string target consistently
		if df[target].dtype != "string":
			df[target] = df[target].astype("string")

		# Mask of rows needing fill
		missing = df[target].isna()

		# Build deterministic mapping for each predictor: group -> the sole category of Outlet_Size
		det_maps = {}
		for p in predictors:
			g = df.loc[~df[target].isna(), [p, target]].groupby(p)[target]
			# groups where Outlet_Size has exactly one unique non-null value
			one_val_groups = g.nunique() == 1
			if one_val_groups.any():
				# extract that single stable value per group
				stable_vals = g.apply(lambda s: s.dropna().iloc[0])
				det_maps[p] = stable_vals[one_val_groups].to_dict()
			else:
				det_maps[p] = {}

		# Sequentially fill using each predictor's deterministic map (never overwrite already-filled)
		for p, mapping in det_maps.items():
			if not mapping:
				continue
			mask = df[target].isna() & df[p].isin(mapping.keys())
			df.loc[mask, target] = df.loc[mask, p].map(mapping).astype(object)

		# Save back
		if(target == "Outlet_Size"):
			self.data[target] = df[target].astype(object)
		else:
			self.data[target] = df[target].astype(float64)

		# Optional: return some quick stats
		return {
			"filled_rows_total": int(missing.sum()) - int(self.data[target].isna().sum()),
			"remaining_nans": int(self.data[target].isna().sum()),
			"deterministic_maps_used": {k: len(v) for k, v in det_maps.items()}
		}

	def count_case_b_candidates(self, target, predictors):
		df = self.data

		report = {}
		missing = df[target].isna()

		for p in predictors:
			# Among rows where target is known, find groups with >1 unique Outlet_Size
			known = df[target].notna()
			nunique_per_group = df.loc[known].groupby(p)[target].nunique()
			mixed_groups = nunique_per_group[nunique_per_group > 1].index.tolist()

			# How many missing rows fall into these mixed groups?
			missing_in_mixed = df.loc[missing & df[p].isin(mixed_groups)]
			report[p] = {
				"num_mixed_groups": len(mixed_groups),
				"mixed_groups_sample": mixed_groups[:10],  # preview
				"missing_rows_in_mixed_groups": int(missing_in_mixed.shape[0]),
			}

		# Also report how many NaNs remain overall for the target
		report["overall"] = {
			f"total_missing_{target}": int(missing.sum())
		}
		return report


	def test_dependency_of_outlet_size(self):

		print("Outlet_Type : ", pd.crosstab(self.data["Outlet_Type"], self.data["Outlet_Size"], normalize="index"))
		print("Outlet_Location_Type : ", pd.crosstab(self.data["Outlet_Location_Type"], self.data["Outlet_Size"], normalize="index"))
		print("Outlet_Establishment_Year : ", pd.crosstab(self.data["Outlet_Establishment_Year"], self.data["Outlet_Size"], normalize="index"))

		contingency = pd.crosstab(self.data["Outlet_Type"], self.data["Outlet_Size"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Outlet_Type):", p)

		contingency = pd.crosstab(self.data["Outlet_Location_Type"], self.data["Outlet_Size"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Outlet_Location_Type):", p)

		contingency = pd.crosstab(self.data["Outlet_Establishment_Year"], self.data["Outlet_Size"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Outlet_Establishment_Year):", p)

	def impute_outlet_size(self):

		target = "Outlet_Size"
		predictors = ["Outlet_Type", "Outlet_Location_Type", "Outlet_Establishment_Year"]
		stats_a = self.impute_case_a(target, predictors)
		print(stats_a)
		print()
		case_b_report = self.count_case_b_candidates(target, predictors)
		print(case_b_report)

	def test_dependency_of_Item_Weight(self):

		print("Item_Fat_Content : ", pd.crosstab(self.data["Item_Fat_Content"], self.data["Item_Weight"], normalize="index"))
		print("Item_Type : ", pd.crosstab(self.data["Item_Type"], self.data["Item_Weight"], normalize="index"))
		print("Item_MRP : ", pd.crosstab(self.data["Item_MRP"], self.data["Item_Weight"], normalize="index"))

		contingency = pd.crosstab(self.data["Item_Fat_Content"], self.data["Item_Weight"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Item_Fat_Content):", p)

		contingency = pd.crosstab(self.data["Item_Type"], self.data["Item_Weight"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Item_Type):", p)

		contingency = pd.crosstab(self.data["Item_MRP"], self.data["Item_Weight"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Item_MRP):", p)

	def impute_item_weight(self):

		target = "Item_Weight"
		predictors = ["Item_Fat_Content", "Item_Type", "Item_MRP"]
		stats_a = self.impute_case_a(target, predictors)
		print(stats_a)
		print()
		case_b_report = self.count_case_b_candidates(target, predictors)
		print(case_b_report)

	def test_dependency_of_Item_Visibility(self):

		contingency = pd.crosstab(self.data["Item_Fat_Content"], self.data["Item_Visibility"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Item_Fat_Content):", p)

		contingency = pd.crosstab(self.data["Item_Type"], self.data["Item_Visibility"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Item_Type):", p)

		contingency = pd.crosstab(self.data["Item_MRP"], self.data["Item_Visibility"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Item_MRP):", p)

		contingency = pd.crosstab(self.data["Item_Identifier"], self.data["Item_Visibility"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Item_Identifier):", p)

		contingency = pd.crosstab(self.data["Outlet_Identifier"], self.data["Item_Visibility"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Outlet_Identifier):", p)

		contingency = pd.crosstab(self.data["Outlet_Location_Type"], self.data["Item_Visibility"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Outlet_Location_Type):", p)

		contingency = pd.crosstab(self.data["Outlet_Size"], self.data["Item_Visibility"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Outlet_Size):", p)

		contingency = pd.crosstab(self.data["Outlet_Type"], self.data["Item_Visibility"])
		chi2, p, _, _ = chi2_contingency(contingency)
		print("p-value (Outlet_Type):", p)

	def impute_item_visibility(self):

		target = "Item_Visibility"
		predictors = ["Item_Identifier", "Outlet_Identifier"]
		stats_a = self.impute_case_a(target, predictors)
		print(stats_a)
		print()
		case_b_report = self.count_case_b_candidates(target, predictors)
		print(case_b_report)
		self.impute_item_visibility_by_pair()

	def impute_item_visibility_by_pair(self, min_pair_count=3):
		"""
		Hierarchical imputation for Item_Visibility:
		  (Item_Identifier, Outlet_Identifier) -> Item_Identifier -> Outlet_Identifier -> Outlet_Type -> global
		Uses medians computed from self.data only. Returns stats dict.
		"""
		df = self.data.copy()
		col = "Item_Visibility"

		if not pd.api.types.is_numeric_dtype(df[col]):
			s = df[col].astype("string").str.strip()
			# drop thousands separators and blank tokens
			s = s.str.replace(",", "", regex=False)
			s = s.replace({"": pd.NA, "NA": pd.NA, "NaN": pd.NA, "nan": pd.NA, None: pd.NA})
			df[col] = pd.to_numeric(s, errors="coerce")
		# treat zeros as missing (typical for this dataset)
		df[col] = df[col].replace(0, nan).astype(float64)


		# --- build TRAIN stats only ---
		# Pair-level medians (require min_pair_count to avoid noisy 1-offs)
		pair_counts = (df[df[col].notna()]
					   .groupby(["Item_Identifier", "Outlet_Identifier"])[col]
					   .size())
		valid_pairs = pair_counts[pair_counts >= min_pair_count].index
		pair_median = (df[df[col].notna()]
					   .groupby(["Item_Identifier", "Outlet_Identifier"])[col]
					   .median())
		# keep only pairs with enough support
		pair_median = pair_median.loc[pair_median.index.isin(valid_pairs)]
		pair_dict = pair_median.to_dict()

		# Item-level median
		item_median = (df[df[col].notna()]
					   .groupby("Item_Identifier")[col]
					   .median())
		item_dict = item_median.to_dict()

		# Outlet-level median
		outlet_median = (df[df[col].notna()]
						 .groupby("Outlet_Identifier")[col]
						 .median())
		outlet_dict = outlet_median.to_dict()

		# Outlet_Type median (borderline useful per your p-value)
		otype_median = (df[df[col].notna()]
						.groupby("Outlet_Type")[col]
						.median())
		otype_dict = otype_median.to_dict()

		global_med = float(df[col].median())

		# --- apply to target df in-place order: pair -> item -> outlet -> outlet_type -> global ---
		miss_before = int(df[col].isna().sum())

		# 1) Pair
		pairs = list(zip(df["Item_Identifier"], df["Outlet_Identifier"]))
		fill1 = pd.Series(pairs, index=df.index).map(pair_dict)
		df[col] = df[col].fillna(fill1).astype(float64)
		df[col] = df[col].fillna(df["Outlet_Identifier"].map(outlet_dict)).astype(float64)
		# df[col] = df[col].fillna(df["Item_Identifier"].map(item_dict))
		# df[col] = df[col].fillna(df["Outlet_Type"].map(otype_dict))

		# ensure type
		df[col] = df[col].astype(float64)

		self.data[col] = df[col].astype(float64)
		stats = {
			"missing_before": miss_before,
			"missing_after": int(df[col].isna().sum()),
			"filled_count": int(miss_before - df[col].isna().sum()),
			"used_pairs": len(pair_dict),
			"used_items": len(item_dict),
			"used_outlets": len(outlet_dict),
			"global_median": global_med,
		}
		print(stats)
		return stats

	def correct_item_fat_content_values(self):

		self.data["Item_Fat_Content"] = self.data["Item_Fat_Content"].replace("LF", "Low Fat")
		self.data["Item_Fat_Content"] = self.data["Item_Fat_Content"].replace("low fat", "Low Fat")
		self.data["Item_Fat_Content"] = self.data["Item_Fat_Content"].replace("reg", "Regular")

		m = self.data["Item_Identifier"].astype(object).str.startswith("NC", na=False)
		self.data.loc[m, "Item_Fat_Content"] = "Inedible"

	def convert_int_to_float(self):

		columns = self.data.columns
		columnType = [type(self.data[c][1]) for c in columns]
		columnTypeSet = set(columnType)
		print("The data types present in the train set are : ", columnTypeSet)

		typeToChoose = float32
		for col in self.data.select_dtypes(include=[float64]).columns:
			diff = (self.data[col] - self.data[col].astype('float32')).abs()
			max_diff = diff.max()
			if(max_diff> 1e-5):
				typeToChoose = float64
			# print(f"{col}: max absolute difference = {max_diff}")

		for c, ctype in zip(columns, columnType):
			if(ctype == int64):
				self.data[c] = self.data[c].astype(typeToChoose)
		print("Converted int columns to :", typeToChoose)

		columnType = [type(self.data[c][1]) for c in columns]
		columnTypeSet = set(columnType)
		print("The data types present in the train set are : ", columnTypeSet)

	def encode_str_to_float(self):
		missingTokens = ("", " ", "na", "n/a", "none", "null", "-", "NaN", "NA", "NULL")

		strColumns = list(self.data.select_dtypes(include=[object, "string"]).columns)
		print(strColumns)

		categoricalDict = {}
		forwardMaps = {}
		reverseMaps = {}

		self.data = deepcopy(self.data)
		uniqueDict = {}
		uniqueCountDict = {}
		for col in strColumns:
			s = self.data[col]
			s = s.astype(str)
			s = s.str.strip()
			uniqueDict[col] = sorted(self.data[col].unique())
			uniqueCountDict[col] = self.data[col].nunique()
			if(col == "Item_Identifier"):
				continue
			print(col ,"\t", uniqueDict[col])
			print(col ,"\t", uniqueCountDict[col])

		for col in strColumns:
			size = uniqueCountDict[col]
			replaceList = [i for i in range(size)]
			map_ = {entry:replacement for entry,replacement in zip(uniqueDict[col], replaceList)}
			self.data[col] = self.data[col].replace(map_).astype(float64)

		columns = self.data.columns
		columnType = [type(self.data[c][1]) for c in columns]
		columnTypeSet = set(columnType)
		print("The data types present in the train set are : ", columnTypeSet)
		print()

		uniqueDict = {}
		uniqueCountDict = {}
		for col in strColumns:
			s = self.data[col]
			s = s.astype(str)
			s = s.str.strip()
			uniqueDict[col] = sorted(self.data[col].unique())
			if(nan in uniqueDict[col]):
				uniqueDict[col].remove(nan)
				uniqueCountDict[col] = self.data[col].nunique() - 1
			else:
				uniqueCountDict[col] = self.data[col].nunique()
			if(col == "Item_Identifier"):
				continue
			print(col ,"\t", uniqueDict[col])
			print(col ,"\t", uniqueCountDict[col])

	def process_data(self, type_="train"):
		self.processType = type_
		self.read_data()
		self.convert_zero_visibility_to_na()
		print("\nInitial NAs")
		self.count_na()

		self.fill_mappable_na()
		print("\nAfter Filling mappable NAs")
		self.count_na()
		
		# self.test_dependency_of_outlet_size()
		self.impute_outlet_size()
		print("\nAfter Imputing Outlet Size")
		self.count_na()
		
		# self.test_dependency_of_Item_Weight()
		self.impute_item_weight()
		print("\nAfter Imputing Item Weight")
		self.count_na()
		
		m = self.data["Item_Weight"].mode(dropna=True)
		self.data["Item_Weight"] = self.data["Item_Weight"].fillna(m.iloc[0])
		print("\nAfter Dropping Item Weight")
		self.count_na()

		# self.test_dependency_of_Item_Visibility()
		self.impute_item_visibility()
		print("\nAfter Imputing Item Visibility")
		self.count_na()

		self.correct_item_fat_content_values()
		
		self.convert_int_to_float()
		self.encode_str_to_float()

		if(self.processType == "train"):
			self.data.to_csv(os.path.join("..", "output", "clean_train.csv"))
		else:
			self.data.to_csv(os.path.join("..", "output", "clean_test.csv"))

if __name__ == "__main__":
	print("Start.")
	w = Wrangler()
	w.process_data(type_="train")
	w.process_data(type_="test")


	print("End.")