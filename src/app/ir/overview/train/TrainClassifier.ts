import { TrainClassifierOptions } from '@/nodes/overview/train/TrainClassifier';
import { nodeName } from '@/app/ir/irCommon';

export default class TrainClassifier {
  constructor(
    public readonly name: string,
    public readonly LossFunction: string,
    public readonly Epochs: number,
    public readonly Device: string,
    public readonly LogInterval: number,
  ) {
  }

  static build(options: Map<string, any>): TrainClassifier {
    return new TrainClassifier(
      options.get(nodeName),
      options.get(TrainClassifierOptions.LossFunction),
      options.get(TrainClassifierOptions.Epochs),
      options.get(TrainClassifierOptions.Device),
      options.get(TrainClassifierOptions.LogInterval),
    );
  }

  public initCode(): string[] {
    return [`
def train_classifier(model, train_loader, test_loader, optimizer, loss, device, epoch, log_interval=${this.LogInterval}):
  def train():
    model.train()
    running_loss = 0
    for i, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      if log_interval > 0 and i % log_interval == log_interval - 1:
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / log_interval))
        running_loss = 0.0

  def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\\nTest set: Average loss: {:.4f}\\n'.format(test_loss))

  for epoch in range(${this.Epochs}):
    train()
    test()
`.trim()];
  }

  public callCode(params: string[]): string[] {
    return [`train_classifier(${params.join(', ')}, device="${this.Device}", epoch=${this.Epochs}, log_interval=${this.LogInterval})`];
  }
}
