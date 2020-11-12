import istateToGraph from '@/app/ir/istateToGraph';
import InModel from '@/app/ir/InModel';
import Conv2D from '@/app/ir/model/conv2d';

describe('Graph', () => {
  const graph1 = '{"nodes":[{"type":"InModel","id":"node_16039892032880","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16039892032881","value":null}]],"position":{"x":21,"y":234},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16039892157612","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_16039892157623","value":null}],["Output",{"id":"ni_16039892157624","value":null}]],"position":{"x":312,"y":230},"width":200,"twoColumn":false},{"type":"MaxPool2d","id":"node_16039892316758","name":"MaxPool2d","options":[["Kernel size",[0,0]],["Stride",[0,0]],["Padding",[0,0]],["Dilation",[1,1]],["Return indices",null],["Ceil mode",null]],"state":{},"interfaces":[["Input",{"id":"ni_16039892316759","value":null}],["Output",{"id":"ni_160398923167510","value":null}]],"position":{"x":331,"y":479},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160398924063014","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160398924063015","value":null}]],"position":{"x":608,"y":437},"width":200,"twoColumn":false}],"connections":[{"id":"16039892205757","from":"ni_16039892032881","to":"ni_16039892157623"},{"id":"160398923656513","from":"ni_16039892157624","to":"ni_16039892316759"},{"id":"160398961820424","from":"ni_160398923167510","to":"ni_160398924063015"}],"panning":{"x":0,"y":0},"scaling":1}';
  const graph2 = '{"nodes":[{"type":"InModel","id":"node_16039996870190","name":"InModel","options":[],"state":{},"interfaces":[["Output",{"id":"ni_16039996870191","value":null}]],"position":{"x":-77.46078244707978,"y":106},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_16039996941842","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_16039996941853","value":null}],["Output",{"id":"ni_16039996941854","value":null}]],"position":{"x":186.17647218821736,"y":115.49142073530265},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160400104763047","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_160400104763048","value":null}],["Output",{"id":"ni_160400104763149","value":null}]],"position":{"x":503.7178227039244,"y":59.36506071686672},"width":200,"twoColumn":false},{"type":"Conv2d","id":"node_160400110066953","name":"Conv2d","options":[["In channels",0],["Out channels",0],["Kernel size",[0,0]],["Stride",[1,1]],["Padding",[0,0]],["Dilation",[1,1]],["Groups",1],["Bias",1],["Padding mode","zeros"]],"state":{},"interfaces":[["Input",{"id":"ni_160400110066954","value":null}],["Output",{"id":"ni_160400110066955","value":null}]],"position":{"x":508.5378485872821,"y":288.9421610499153},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160400112448859","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400112448860","value":null}]],"position":{"x":911.0666462044529,"y":29.87844102632479},"width":200,"twoColumn":false},{"type":"OutModel","id":"node_160400113171465","name":"OutModel","options":[],"state":{},"interfaces":[["Input",{"id":"ni_160400113171466","value":null}]],"position":{"x":939.5004691338713,"y":306.3183861734489},"width":200,"twoColumn":false}],"connections":[{"id":"16039996976097","from":"ni_16039996870191","to":"ni_16039996941853"},{"id":"160400105094252","from":"ni_16039996941854","to":"ni_160400104763048"},{"id":"160400110318058","from":"ni_16039996941854","to":"ni_160400110066954"},{"id":"160400112847964","from":"ni_160400104763149","to":"ni_160400112448860"},{"id":"160400641665775","from":"ni_160400110066955","to":"ni_160400113171466"}],"panning":{"x":177.3940230694035,"y":189.50554993466366},"scaling":0.6172136465643434}';
  const istate1 = JSON.parse(graph1);
  const istate2 = JSON.parse(graph2);

  it('t1', () => {
    const graph1 = istateToGraph(istate1);
    const initial = graph1.nodesAsArray.find((n) => n.mlNode instanceof InModel);
    if (initial === undefined) throw new Error();
    const next = graph1.nextNodesFrom(initial)[0];
    expect(next.mlNode).toBeInstanceOf(Conv2D);
  });

  it('bifurcations', () => {
    const graph = istateToGraph(istate2);
    const initial = graph.nodesAsArray.find((n) => n.mlNode instanceof InModel)!;
    const second = graph.nextNodesFrom(initial)[0];
    expect(second.mlNode).toBeInstanceOf(Conv2D);
    expect(second.outputInterfaces.size).toBe(1);
    expect(graph.nextNodesFrom(second).length).toBeGreaterThan(1);
  });
});
