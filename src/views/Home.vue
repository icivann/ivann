<template>
  <div class="home">
    <NodeEditor/>
  </div>
</template>

<script>
// @ is an alias to /src
import NodeEditor from '@/components/NodeEditor.vue';

export default {
  name: 'Home',
  components: {
    NodeEditor,
  },
};
</script>
