import GraphNode from '@/app/ir/GraphNode';

const prefix = 'import tensorflow as tf\n'
             + 'import tensorflow.keras as keras\n'
             + 'import tensorflow.keras.layers as layers\n';

const suffix = 'model.compile(\'adam\', \'mean_squared_error\')\n';

function generateKeras(nodes: GraphNode[]): string {
  return prefix + nodes.map((it) => it.mlNode.code()).join('\n') + suffix;
}

export default generateKeras;
