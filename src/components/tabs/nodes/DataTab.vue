<template>
  <NodesTab :search-items="searchItems"/>
</template>

<script lang="ts">
import Vue from 'vue';
import { Component } from 'vue-property-decorator';
import { convertToSearch, modify } from '@/components/SearchUtils';
import EditorManager from '@/EditorManager';
import { DataCategories, DataNodes } from '@/nodes/data/Types';
import NodesTab from './NodesTab.vue';

@Component({
  components: { NodesTab },
})
export default class DataTab extends Vue {
  private searchItems = convertToSearch(EditorManager.getInstance().dataCanvas.nodeList);

  created() {
    this.searchItems = modify(this.searchItems, DataCategories.IO, DataNodes.InData, {
      name: DataNodes.InData,
      displayName: 'Input',
    });
  }
}
</script>

<style scoped>

</style>
