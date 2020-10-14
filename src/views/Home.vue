<template>
  <div class="home">
    <Navbar />
    <NodeEditor/>
  </div>
</template>

<script>
// @ is an alias to /src
import NodeEditor from '@/components/NodeEditor.vue';
import Navbar from '@/components/Navbar.vue';

export default {
  name: 'Home',
  components: {
    Navbar,
    NodeEditor,
  },
};
</script>
