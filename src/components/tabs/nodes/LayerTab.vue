<template>
  <NodesTab :search-items="searchItems"/>
</template>

<script lang="ts">
import Vue from 'vue';
import { Component } from 'vue-property-decorator';
import { ModelCategories, ModelNodes } from '@/nodes/model/Types';
import { convertToSearch, modify } from '@/components/SearchUtils';
import EditorManager from '@/EditorManager';
import NodesTab from './NodesTab.vue';

@Component({
  components: { NodesTab },
})
export default class LayerTab extends Vue {
  private searchItems = convertToSearch(EditorManager.getInstance().modelCanvas.nodeList);

  created() {
    this.searchItems = modify(this.searchItems, ModelCategories.IO, ModelNodes.InModel, {
      name: ModelNodes.InModel,
      displayName: 'Input',
    });
    this.searchItems = modify(this.searchItems, ModelCategories.IO, ModelNodes.OutModel, {
      name: ModelNodes.OutModel,
      displayName: 'Output',
    });
  }
}
</script>

<style scoped>

</style>
