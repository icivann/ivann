<template>
  <div class="layers-tab h-100">
    <ExpandablePanel name="Core">
      This contains core layer nodes
    </ExpandablePanel>
    <ExpandablePanel name="Convolutional">
      This contains convolution layers
    </ExpandablePanel>
    <ExpandablePanel name="Pooling">
      This contains pooling layer nodes (obvs).
    </ExpandablePanel>
  </div>
</template>

<script>
import { Component, Vue } from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';

@Component({
  components: { ExpandablePanel },
})
export default class LayersTab extends Vue {
}
</script>

<style scoped>
</style>
