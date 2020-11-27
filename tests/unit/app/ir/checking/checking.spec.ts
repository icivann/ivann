import istateToGraph from '@/app/ir/istateToGraph';
import { check } from '@/app/ir/checking/check';
import Graph from '@/app/ir/Graph';
import IrError from '@/app/ir/checking/irError';
import Conv2d from '@/app/ir/model/conv2d';
import Conv1d from '@/app/ir/model/conv1d';
import Linear from '@/app/ir/model/linear';

function expectError(graph: Graph, should: (err: IrError) => void) {
  const errors = check(graph);
  expect(errors.length).toBeGreaterThan(0);
  for (const error of errors) {
    should(error);
  }
}

function expectNoErrors(graph: Graph) {
  const errors = check(graph);
  expect(errors.length).toBe(0);
}

describe('type checking', () => {
  const outputOrphan = istateToGraph(
    JSON.parse('{"nodes":[{"type":"InModel","id":"node_16040740027440","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16040740027441","value":null}]],"position":{"x":106.66666666666669,"y":257.6666666666667},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16040740160792","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_16040740160803","value":null}],["Output",{"id":"ni_16040740160804","value":null}]],"position":{"x":394.6666666666667,"y":341.6666666666667},"width":200,"twoColumn":false}],"connections":[{"id":"16040740240767","from":"ni_16040740027441","to":"ni_16040740160803"}],"panning":{"x":0,"y":0},"scaling":1}'),
  );

  it('orphans report errors', () => {
    const second = outputOrphan.nodesAsArray.filter((n) => n.mlNode instanceof Conv2d)[0];
    expectError(outputOrphan, (err) => {
      expect(err.offenders).toContain(second);
    });
  });

  it('bad convoliution channels report errors', () => {
    const graph = istateToGraph(
      JSON.parse('{"nodes":[{"type":"InModel","id":"node_16062528273010","name":"i","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16062528273011","value":null}]],"position":{"x":78,"y":158},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_16062528318402","name":"o","options":[],"state":{},"interfaces":[["Input",{"id":"ni_16062528318413","value":null}]],"position":{"x":765,"y":197},"width":200,"twoColumn":false},{"type":"Conv1d","id":"node_160625823243814","name":"Conv1d YOLO","options":[["In channels",0],["Out channels",4],["Kernel size",[0]],["Stride",[2]],["Padding",[0]],["Dilation",[1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_160625823243815","value":null}],["Output",{"id":"ni_160625823243816","value":null}]],"position":{"x":206,"y":420},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160625823380817","name":"Conv2d","options":[["In channels",0],["Out channels",3],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_160625823380818","value":null}],["Output",{"id":"ni_160625823380819","value":null}]],"position":{"x":553,"y":420},"width":200,"twoColumn":false}],"connections":[{"id":"160625823620822","from":"ni_16062528273011","to":"ni_160625823243815"},{"id":"160625823870725","from":"ni_160625823243816","to":"ni_160625823380818"},{"id":"160625824086728","from":"ni_160625823380819","to":"ni_16062528318413"}],"panning":{"x":1.4926523584755387,"y":1.657796874732412},"scaling":0.9968341980495156}'),
    );
    const conv2d = graph.nodesAsArray.filter((n) => n.mlNode instanceof Conv2d)[0];
    const conv1d = graph.nodesAsArray.filter((n) => n.mlNode instanceof Conv1d)[0];
    expectError(graph, (err) => {
      expect(err.offenders.length).toBe(2);
      expect(err.offenders).toContain(conv1d);
      expect(err.offenders).toContain(conv2d);
    });
  });

  it('bad linear layers\' features report errors', () => {
    const graph = istateToGraph(
      JSON.parse('{"nodes":[{"type":"Linear","id":"node_16064076698330","name":"Linear","options":[["In features",2],["Out features",0],["Bias",1]],"state":{},"interfaces":[["Input",{"id":"ni_16064076698331","value":null}],["Output",{"id":"ni_16064076698332","value":null}]],"position":{"x":271,"y":93},"width":200,"twoColumn":false},{"type":"Linear","id":"node_16064076710283","name":"Linear","options":[["In features",4],["Out features",0],["Bias",1]],"state":{},"interfaces":[["Input",{"id":"ni_16064076710284","value":null}],["Output",{"id":"ni_16064076710285","value":null}]],"position":{"x":551,"y":88},"width":200,"twoColumn":false},{"type":"InModel","id":"node_16064077189139","name":"i","options":[],"state":{},"interfaces":[["Output",{"id":"ni_160640771891310","value":null}]],"position":{"x":30,"y":221},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160640772095011","name":"o","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160640772095012","value":null}]],"position":{"x":803,"y":153},"width":200,"twoColumn":false}],"connections":[{"id":"160640772450116","from":"ni_160640771891310","to":"ni_16064076698331"},{"id":"160640772980719","from":"ni_16064076698332","to":"ni_16064076710284"},{"id":"160640773190522","from":"ni_16064076710285","to":"ni_160640772095012"}],"panning":{"x":0,"y":0},"scaling":1}'),
    );
    const linearNodes = graph.nodesAsArray.filter((n) => n.mlNode instanceof Linear);
    const linear1 = linearNodes[0];
    const linear2 = linearNodes[1];
    expectError(graph, (err) => {
      expect(err.offenders.length).toBe(2);
      expect(err.offenders).toContain(linear1);
      expect(err.offenders).toContain(linear2);
    });
  });
});
