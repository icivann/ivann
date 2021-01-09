export enum OverviewCategories {
  Model = 'Model',
  Data = 'Data',
  Custom = 'Custom',
  Train = 'Train',
  Optimizer = 'Optimizer',
  LossFunctions = 'Loss'
}

export enum OverviewNodes {
  ModelNode = 'ModelNode',
  DataNode = 'DataNode',
  TrainClassifier = 'TrainClassifier',
  Adadelta = 'Adadelta',
  OverviewCustom = 'OverviewCustom',
  // loss functions
  NLLLoss= 'NLLLoss',
  MultiMarginLoss = 'MultiMarginLoss',
  MultiLabelSoftMarginLoss = 'MultiLabelSoftMarginLoss',
  MultiLabelMarginLoss = 'MultiLabelMarginLoss',
  MSELoss = 'MSELoss',
  L1Loss= 'L1Loss',
  KLDivLoss = 'KLDivLoss',
  HingeEmbeddingLoss = 'HingeEmbeddingLoss',
  EmbeddingBag = 'EmbeddingBag',
  CTCLoss = 'CTCLoss',
  CrossEntropyLoss = 'CrossEntropyLoss',
  CosineEmbeddingLoss = 'CosineEmbeddingLoss',
  BCEWithLogitsLoss = 'BCEWithLogitsLoss',
  BCELoss = 'BCELoss',
  MarginRankingLoss = 'MarginRankingLoss',
  TripletMarginLoss = 'TripletMarginLoss',
  SmoothL1Loss= 'SmoothL1Loss',
  PoissonNLLLoss = 'PoissonNLLLoss',
}
