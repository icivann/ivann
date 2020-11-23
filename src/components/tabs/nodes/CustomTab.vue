<template>
  <div>
    <div
      class="msg"
      v-if="files.length === 0"
      @click="clickCodeVault"
    >
      Click Here to Add Custom Functions
    </div>
    <NodesTab v-else :searchItems="searchItems"/>
  </div>
</template>

<script lang="ts">
import { Component, Vue, Watch } from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import { Getter, Mutation } from 'vuex-class';
import NodesTab from '@/components/tabs/nodes/NodesTab.vue';
import { ParsedFile } from '@/store/codeVault/types';
import { SearchItem } from '@/components/SearchUtils';
import { CommonNodes } from '@/nodes/common/Types';

@Component({
  components: {
    NodesTab,
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
})
export default class CustomTab extends Vue {
  private searchItems!: SearchItem[];
  @Mutation('enterCodeVault') enterCodeVault!: () => void;
  @Mutation('closeFiles') closeFiles!: () => void;
  @Getter('files') files!: ParsedFile[];

  created() {
    this.updateFiles(this.files);
  }

  @Watch('files')
  private updateFiles(files: ParsedFile[]) {
    console.log('Updated');
    this.searchItems = files.map((file) => ({
      category: file.filename,
      nodes: file.functions.map((func) => ({
        name: CommonNodes.Custom,
        displayName: func.name,
        options: func,
      })),
    }));
  }

  private clickCodeVault() {
    this.closeFiles();
    this.enterCodeVault();
  }
}
</script>

<style scoped>
  .msg {
    text-align: center;
    background: var(--background);
    border-radius: 4px;
    border-style: solid;
    border-width: 1px;
    margin-top: 5px;
    border-color: var(--grey);
    padding: 10px;
  }

  .msg:hover {
    background: #1c1c1c;
    transition-duration: 0.1s;
    cursor: pointer;
    border-color: var(--foreground);
  }
</style>
