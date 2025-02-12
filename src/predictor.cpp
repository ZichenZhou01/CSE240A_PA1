//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include <math.h>
#include "predictor.h"
#include <vector>
#include <cmath>

//
// TODO:Student Information
//
const char *studentName = "Ash Zhou";
const char *studentID = "A16393178";
const char *email = "ziz022@ucsd.edu";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = {"Static", "Gshare",
                         "Tournament", "Custom"};

// define number of bits required for indexing the BHT here.
int ghistoryBits = 16; // Number of bits used for Global History
int bpType;            // Branch Prediction Type
int verbose;
int lhistoryBits = 10;
int pcIndexBits = 12;
int choiceBits = 10;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//

//
// TODO: Add your own Branch Predictor data structures here
uint8_t *global_BHT;
uint32_t *local_PHT;
uint8_t *local_BHT;
uint8_t *choice_BHT;
uint32_t global_history;
//
// gshare
uint8_t *bht_gshare;
uint64_t ghistory;

int ghistoryBits_per = 11;
int perceptron_power = 13;
int number_perceptron = -1;
int threshold = -1;
std::vector< std::vector<int8_t> > perceptron_weights;
uint64_t global_history_custom;
//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//

// gshare functions
void init_gshare()
{
  int bht_entries = 1 << ghistoryBits;
  bht_gshare = (uint8_t *)malloc(bht_entries * sizeof(uint8_t));
  int i = 0;
  for (i = 0; i < bht_entries; i++)
  {
    bht_gshare[i] = WN;
  }
  ghistory = 0;
}


void init_tournament() {

  int number_global = 1 << ghistoryBits;
  int number_local_BHT = 1 << lhistoryBits;
  int number_choice = 1 << choiceBits;
  int number_local_PHT = 1 << pcIndexBits;

  global_BHT = (uint8_t*) malloc(number_global * sizeof(uint8_t));
  local_PHT = (uint32_t*) malloc(number_local_PHT * sizeof(uint32_t));
  local_BHT = (uint8_t*) malloc(number_local_BHT * sizeof(uint8_t));
  choice_BHT = (uint8_t*) malloc(number_choice * sizeof(uint8_t));

  for (int i = 0; i < number_global; i++){
    global_BHT[i] = WN;
  }

  for (int i = 0; i < number_local_PHT; i++){
    local_PHT[i] = SN;
  }

  for (int i = 0; i < number_local_BHT; i++){
    local_BHT[i] = WN;
  }

  for (int i = 0; i < number_choice; i++){
    choice_BHT[i] = WG;
  }

  global_history = 0;

}

void init_perceptron(){

  threshold = static_cast<int>(floor(1.93 * ghistoryBits_per + 14));
  number_perceptron = std::pow(2, perceptron_power);
  perceptron_weights.resize(number_perceptron, std::vector<int8_t>(ghistoryBits_per + 1, 0));
  global_history_custom = 0;

}

uint8_t tournament_predict(uint32_t pc) {
  
  uint32_t global_BHT_Ind = global_history & ((1 << ghistoryBits) - 1);
  uint8_t global_preidction = global_BHT[global_BHT_Ind];

  uint32_t local_PHT_Ind = pc & ((1 << pcIndexBits) - 1);
  uint32_t local_BHT_Ind = local_PHT[local_PHT_Ind];
  uint8_t local_prediction = local_BHT[local_BHT_Ind];

  uint32_t choice_BHT_Ind = global_history & ((1 << ghistoryBits) - 1);
  uint8_t choice_prediction = choice_BHT[choice_BHT_Ind];

  uint8_t result = -1;

  if (choice_prediction <= WG){
    if (global_preidction <= WN){
      result = NOTTAKEN;
    } else{
      result = TAKEN;
    }
  } else{
    if (local_prediction <= WN){
      result = NOTTAKEN;
    } else{
      result = TAKEN;
    }
  }

  return result;
}

uint8_t perceptron_predict(uint32_t pc) {

  uint32_t perceptron_entries = 1 << perceptron_power;
  uint32_t pc_lower_bits = pc & (perceptron_entries - 1);
  uint32_t ghistory_lower_bits = global_history_custom & (perceptron_entries - 1);
  uint32_t index = pc_lower_bits ^ ghistory_lower_bits;
  // uint32_t index = pc % number_perceptron;
  int8_t output_weight = perceptron_weights[index][0];
  std::vector<int8_t> x_bits (ghistoryBits_per, 0);

  for (int i = 0; i < ghistoryBits_per; i++){
    if (((global_history_custom >> i) & 1) == 1){
      x_bits[i] = 1;
    } else{
      x_bits[i] = -1;
    }
  }

  for (int i = 0; i < ghistoryBits_per; i++){
    output_weight += perceptron_weights[index][i+1] * x_bits[i];
  }

  uint8_t result = -1;
  if (output_weight < 0){
    result = NOTTAKEN;
  } else{
    result = TAKEN;
  }


  return result;
}

uint8_t gshare_predict(uint32_t pc)
{
  // get lower ghistoryBits of pc
  uint32_t bht_entries = 1 << ghistoryBits;
  uint32_t pc_lower_bits = pc & (bht_entries - 1);
  uint32_t ghistory_lower_bits = ghistory & (bht_entries - 1);
  uint32_t index = pc_lower_bits ^ ghistory_lower_bits;
  switch (bht_gshare[index])
  {
  case WN:
    return NOTTAKEN;
  case SN:
    return NOTTAKEN;
  case WT:
    return TAKEN;
  case ST:
    return TAKEN;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    return NOTTAKEN;
  }
}

void train_gshare(uint32_t pc, uint8_t outcome)
{
  // get lower ghistoryBits of pc
  uint32_t bht_entries = 1 << ghistoryBits;
  uint32_t pc_lower_bits = pc & (bht_entries - 1);
  uint32_t ghistory_lower_bits = ghistory & (bht_entries - 1);
  uint32_t index = pc_lower_bits ^ ghistory_lower_bits;

  // Update state of entry in bht based on outcome
  switch (bht_gshare[index])
  {
  case WN:
    bht_gshare[index] = (outcome == TAKEN) ? WT : SN;
    break;
  case SN:
    bht_gshare[index] = (outcome == TAKEN) ? WN : SN;
    break;
  case WT:
    bht_gshare[index] = (outcome == TAKEN) ? ST : WN;
    break;
  case ST:
    bht_gshare[index] = (outcome == TAKEN) ? ST : WT;
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    break;
  }

  // Update history register
  ghistory = ((ghistory << 1) | outcome);
}

void train_tournament(uint32_t pc, uint8_t outcome){

  uint32_t global_BHT_Ind = global_history & ((1 << ghistoryBits) - 1);
  uint8_t global_preidction = global_BHT[global_BHT_Ind];

  uint32_t local_PHT_Ind = pc & ((1 << pcIndexBits) - 1);
  uint32_t local_BHT_Ind = local_PHT[local_PHT_Ind];
  uint8_t local_prediction = local_BHT[local_BHT_Ind];

  uint32_t choice_BHT_Ind = global_history & ((1 << ghistoryBits) - 1);
  uint8_t choice_prediction = choice_BHT[choice_BHT_Ind];

  if (outcome == NOTTAKEN) {
    if (global_preidction > SN){
      global_BHT[global_history]-= 1;
    }

    if (local_prediction > SN){
      local_BHT[local_BHT_Ind]-= 1;
    }
  } else{
    if (global_preidction < ST) {
      global_BHT[global_history]+= 1;
    }

    if (local_prediction < ST){
      local_BHT[local_BHT_Ind]+= 1;
    }
  }

  uint8_t global_result = -1;
  uint8_t local_result = -1;

  if (global_preidction <= WN){
    global_result = NOTTAKEN;
  } else{
    global_result = TAKEN;
  }

  if (local_prediction <= WN){
    local_result = NOTTAKEN;
  } else{
    local_result = TAKEN;
  }

  if (global_result != local_result){
    if (global_result == outcome){
      if (choice_prediction > SG) {
        choice_BHT[choice_BHT_Ind] -= 1;
      }
    } else{
      if (choice_prediction < SL) {
        choice_BHT[choice_BHT_Ind] += 1;
      }
    }
  }

  local_PHT[local_PHT_Ind] = ((local_BHT_Ind << 1) | outcome);
  local_PHT[local_PHT_Ind] = (local_PHT[local_PHT_Ind]) & ((1 << lhistoryBits) - 1);

  global_history = ((global_history << 1)| outcome);
  global_history = (global_history) & ((1 << ghistoryBits) - 1);

}

void train_perceptron(uint32_t pc, uint8_t outcome){

  uint32_t perceptron_entries = 1 << perceptron_power;
  uint32_t pc_lower_bits = pc & (perceptron_entries - 1);
  uint32_t ghistory_lower_bits = global_history_custom & (perceptron_entries - 1);
  uint32_t index = pc_lower_bits ^ ghistory_lower_bits;
  int8_t output_weight = perceptron_weights[index][0];
  std::vector<int8_t> x_bits (ghistoryBits_per, 0);
  int t_value = -1;

  if (outcome == TAKEN){
    t_value = 1;
  }

  for (int i = 0; i < ghistoryBits_per; i++){
    if (((global_history_custom >> i) & 1) == 1){
      x_bits[i] = 1;
    } else{
      x_bits[i] = -1;
    }
  }

  for (int i = 0; i < ghistoryBits_per; i++){
    output_weight += perceptron_weights[index][i+1] * x_bits[i];
  }

  uint8_t result = -1;
  if (output_weight < 0){
    result = NOTTAKEN;
  } else{
    result = TAKEN;
  }

  if (outcome != result || abs(output_weight) <= threshold) {
    perceptron_weights[index][0] += t_value;

    for (int i = 0; i < ghistoryBits_per; i++) {
      perceptron_weights[index][i + 1] += t_value * x_bits[i];   
    }

  }

  global_history_custom = ((global_history_custom << 1)| outcome);
  global_history_custom = (global_history_custom) & ((1 << ghistoryBits_per) - 1);

}

void cleanup_tournament() {

  free(global_BHT);
  free(local_PHT);
  free(local_BHT);
  free(choice_BHT);

}

void cleanup_gshare()
{
  free(bht_gshare);
}

void init_predictor()
{
  switch (bpType)
  {
  case STATIC:
    break;
  case GSHARE:
    init_gshare();
    break;
  case TOURNAMENT:
    init_tournament();
    break;
  case CUSTOM:
    init_perceptron();
    break;
  default:
    break;
  }
}

// Make a prediction for conditional branch instruction at PC 'pc'
// Returning TAKEN indicates a prediction of taken; returning NOTTAKEN
// indicates a prediction of not taken
//
uint32_t make_prediction(uint32_t pc, uint32_t target, uint32_t direct)
{

  // Make a prediction based on the bpType
  switch (bpType)
  {
  case STATIC:
    return TAKEN;
  case GSHARE:
    return gshare_predict(pc);
  case TOURNAMENT:
    return tournament_predict(pc);
  case CUSTOM:
    return perceptron_predict(pc);
  default:
    break;
  }

  // If there is not a compatable bpType then return NOTTAKEN
  return NOTTAKEN;
}

// Train the predictor the last executed branch at PC 'pc' and with
// outcome 'outcome' (true indicates that the branch was taken, false
// indicates that the branch was not taken)
//

void train_predictor(uint32_t pc, uint32_t target, uint32_t outcome, uint32_t condition, uint32_t call, uint32_t ret, uint32_t direct)
{
  if (condition)
  {
    switch (bpType)
    {
    case STATIC:
      return;
    case GSHARE:
      return train_gshare(pc, outcome);
    case TOURNAMENT:
      return train_tournament(pc, outcome);
    case CUSTOM:
      return train_perceptron(pc, outcome);
    default:
      break;
    }
  }
}
