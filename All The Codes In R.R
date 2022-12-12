library(dplyr) 
library(tidymodels)
library(tidyverse)
library(ggplot2)
library(corrplot)
library(janitor)
library(ISLR)
library(rpart.plot)
library(vip)
library(randomForest)
library(xgboost)
library(parsnip)
library(ranger)
library(tibble)
library(ggcorrplot)
library(caret)
library(klaR)
library(pROC)
library(discrim)
library(pROC)
tidymodels_prefer()
set.seed(0213)

applications_raw = read.csv("data/Application_Data.csv")
applications_raw = applications_raw %>% clean_names()
head(applications_raw)
dim(applications_raw)

summary(applications_raw)

applications = applications_raw %>% 
  select(-applicant_id, -owned_mobile_phone)
head(applications)

applications = applications %>% mutate(status = ifelse(total_bad_debt/(total_bad_debt+total_good_debt) < 0.15, 1, 0)) %>% 
  select(-total_bad_debt,-total_good_debt)

applications = applications %>% mutate(applicant_gender = factor(applicant_gender), owned_car  = factor(owned_car), owned_realty = factor(owned_realty), income_type = factor(income_type), education_type = factor(education_type), family_status = factor(family_status), housing_type =factor(housing_type), owned_work_phone = factor(owned_work_phone), owned_phone =factor(owned_phone), owned_email = factor(owned_email), job_title = factor(job_title), status = factor(status))

head(applications)

summary(applications)
dim(applications)

applications_split = initial_split(applications, prop = 0.7, strata = status)
applications_train = training(applications_split)
applications_test = testing(applications_split)


dim(applications_train)
dim(applications_test)
dim(applications_train)[1] / dim(applications)[1]
dim(applications_train) + dim(applications_test) 

applications_train %>% ggplot(aes(status)) + geom_bar() + labs(title = "Status Count in Training Set", x = "Status", y = "Count") + coord_flip()

applications_train %>% 
  group_by(status,applicant_gender) %>% count %>%
  ggplot(aes(status, n)) + geom_col(show.legend = FALSE) + facet_wrap(~applicant_gender) + labs(title = "Status by Gender", y = "Count", x = "Gender")

applications_train %>% 
  ggplot(aes(status)) +
  geom_histogram(stat="count") +
  facet_wrap(~housing_type) +
  labs(
    title = "Status by housing_type")

applications_train %>% 
  filter(status %in% c(0,1)) %>%
  ggplot(aes(status)) +
  geom_histogram(stat="count") +
  facet_wrap(~housing_type, scales = "free_y") +
  labs(
    title = "Status by housing_type")

ggplot(applications_train, aes(applicant_age)) +
  geom_histogram()

applications_train %>% 
  select(is.numeric) %>% 
  cor() %>% 
  corrplot(type = "lower",  method = 'color', tl.cex = 0.5)

applications_fold = vfold_cv(data = applications_train, v = 5, strata = status)

applications_recipe = recipe(status~ applicant_gender+owned_car+owned_realty+total_children+total_income+income_type+education_type+family_status+housing_type+owned_work_phone+owned_phone+owned_email+job_title+total_family_members+applicant_age+years_of_working, applications_train) %>%
  step_impute_linear(total_children, impute_with = imp_vars(total_family_members)) %>%
  step_dummy(all_nominal_predictors())  %>%
  step_normalize(all_predictors())

log_reg = logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

log_wflow = workflow() %>% 
  add_model(log_reg) %>% 
  add_recipe(applications_recipe)

log_fit = fit_resamples(resamples = applications_fold,log_wflow, control = control_resamples(save_pred = TRUE))

lda = discrim_linear() %>%
  set_mode("classification") %>% 
  set_engine("MASS")

lda_wflow = workflow() %>% 
  add_model(lda) %>% 
  add_recipe(applications_recipe)

lda_fit = fit_resamples(resamples = applications_fold,lda_wflow, control = control_resamples(save_pred = TRUE))

collect_metrics(log_fit)
collect_metrics(lda_fit)

log_fit_a = fit(log_wflow, applications_train)

log_predict = predict(log_fit_a, new_data = applications_test, type = "class")  %>% 
  bind_cols(applications_test %>% select(status)) %>% 
  accuracy(truth = status, estimate = .pred_class)

lda_fit_a = fit(lda_wflow, applications_train)

lda_predict = predict(lda_fit_a, new_data = applications_test, type = "class")  %>% 
  bind_cols(applications_test %>% select(status)) %>% 
  accuracy(truth = status, estimate = .pred_class)

elastic = multinom_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

elastic_workflow = workflow() %>%
  add_recipe(applications_recipe) %>%
  add_model(elastic)

elastic_grid = grid_regular(penalty(range = c(-5, 5)), mixture(range = c(0,1)), levels = 10)

tune_res = tune_grid(
  elastic_workflow,
  resamples = applications_fold,
  grid = elastic_grid,
  metrics = metric_set(roc_auc)
)
autoplot(tune_res)

best = select_best(tune_res)
elastic_final_wflow = finalize_workflow(elastic_workflow, best)
elastic_final_fit = fit(elastic_final_wflow, data = applications_train)

elastic_predict = predict(elastic_final_fit, new_data = applications_test, type = "class")  %>% 
  bind_cols(applications_test %>% select(status)) %>% 
  accuracy(truth = status, estimate = .pred_class)

elastic_ROCAUC = augment(elastic_final_fit, new_data = applications_test) %>%
  select(status, starts_with(".pred"))

tree_spec = rand_forest() %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

class_tree_fit = tree_spec %>%
  fit(status~ applicant_gender+owned_car+owned_realty+total_children+total_income+income_type+education_type+family_status+housing_type+owned_work_phone+owned_phone+owned_email+job_title+total_family_members+applicant_age+years_of_working, data = applications_train)

class_tree_wf = workflow() %>%
  add_model(tree_spec %>% set_args(mtry = tune(), trees = tune(), min_n = tune())) %>%
  add_formula(status~ applicant_gender+owned_car+owned_realty+total_children+total_income+income_type+education_type+family_status+housing_type+owned_work_phone+owned_phone+owned_email+job_title+total_family_members+applicant_age+years_of_working)

regular_grid = grid_regular(mtry(range = c(1,16)), trees(range = c(1,500)), min_n(range = c(1,30)), levels = 4)

tree_res = tune_grid(
  class_tree_wf, 
  resamples = applications_fold, 
  grid = regular_grid, 
  metrics = metric_set(roc_auc)
)

autoplot(tree_res)

tune_regular %>% 
  collect_metrics() %>%
  arrange(by_group = desc(mean))

class_tree_fit %>%
  extract_fit_engine() %>%
  vip()

best_trees = select_best(tune_regular)

tree_forest_final = finalize_workflow(class_tree_wf, best_trees)

tree_forest_final_fit = fit(tree_forest_final, data = applications_train)

tree_forest_predict = predict(tree_forest_final_fit, new_data = applications_test, type = "class")  %>% 
  bind_cols(applications_test %>% select(status)) %>% 
  accuracy(truth = status, estimate = .pred_class)

tree_forest_ROCAUC = augment(tree_forest_final_fit, new_data = applications_test) %>%
  select(status, starts_with(".pred"))

boost_spec = boost_tree(tree_depth = 4) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

boost_fit = fit(boost_spec, status~ applicant_gender+owned_car+owned_realty+total_children+total_income+income_type+education_type+family_status+housing_type+owned_work_phone+owned_phone+owned_email+job_title+total_family_members+applicant_age+years_of_working, data = applications_train)

boost_wf = workflow() %>%
  add_model(boost_spec %>% set_args(trees = tune())) %>%
  add_formula(status~ applicant_gender+owned_car+owned_realty+total_children+total_income+income_type+education_type+family_status+housing_type+owned_work_phone+owned_phone+owned_email+job_title+total_family_members+applicant_age+years_of_working)

boost_grid = grid_regular(trees(range = c(10,400)), levels = 5)

tune_boosted = tune_grid(
  boost_wf,
  resamples = applications_fold,
  grid = boost_grid,
  metrics = metric_set(roc_auc)
)
autoplot(tune_boosted)

best_boosted = select_best(tune_boosted)

boosted_trees_final_wf = finalize_workflow(boost_wf, best_boosted)

boosted_trees_final_fit = fit(boosted_trees_final_wf, data = applications_train)

boosted_trees_predict = predict(boosted_trees_final_fit, new_data = applications_test, type = "class")  %>% 
  bind_cols(applications_test %>% select(status)) %>% 
  accuracy(truth = status, estimate = .pred_class)

boost_trees_ROCAUC = augment(boosted_trees_final_fit, new_data = applications_test) %>%
  select(status, starts_with(".pred"))

accuracies = bind_rows(log_predict,lda_predict,elastic_predict,tree_forest_predict, boosted_trees_predict) %>% 
  tibble() %>% 
  mutate(model = c("Logistic Regression", "LDA", "Elastic Net Tuning", "Random Tree Forest", "Boosted Trees")) %>% 
  select(model, .estimate) %>%
  arrange(desc(.estimate))

accuracies

roc_aucs = bind_rows(df = data.frame(.estimate = collect_metrics(log_fit)$mean[2]), df1 = data.frame(.estimate = collect_metrics(lda_fit)$mean[2]),elastic_ROCAUC %>% roc_auc(status, .pred_0),tree_forest_ROCAUC%>% roc_auc(status, .pred_0), boost_trees_ROCAUC %>% roc_auc(status, .pred_0)) %>% 
  tibble() %>% 
  mutate(model = c("Logistic","LDA","Elastic Net Tuning", "Random Tree Forest", "Boosted Trees")) %>% 
  select(model, .estimate) %>%
  arrange(desc(.estimate))

roc_aucs