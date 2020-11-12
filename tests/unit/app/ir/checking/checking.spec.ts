import istateToGraph from '@/app/ir/istateToGraph';
import { check } from '@/app/ir/checking/check';
import Graph from '@/app/ir/Graph';
import IrError from '@/app/ir/checking/irError';
import Conv2d from '@/app/ir/model/conv2d';

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
});
