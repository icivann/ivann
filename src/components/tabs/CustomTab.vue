<template>
  <div>
    <div
      class="msg"
      v-if="files.length === 0"
      @click="enterCodeVault"
    >
      Click Here to Add Custom Functions
    </div>
    <SearchBar v-else @value-change="search"/>
    <ExpandablePanel
      v-for="(file) in renderedFiles"
      :key="file.filename"
      :name="file.filename"
      v-show="searchString === '' || file.functions.length > 0"
    >
      <ButtonGrid>
        <AddNodeButton
          v-for="(func) in file.functions"
          :key="func.name"
          :node="customNode"
          :name="func.name"
          :options="func"
        />
      </ButtonGrid>
    </ExpandablePanel>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import { CommonNodes } from '@/nodes/common/Types';
import { Getter, Mutation } from 'vuex-class';
import SearchBar from '@/SearchBar.vue';
import { ParsedFile } from '@/store/codeVault/types';
import { OverviewNodes } from '@/nodes/overview/Types';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
    SearchBar,
  },
})
export default class CustomTab extends Vue {
  @Prop({ default: CommonNodes.Custom }) customNode: string = OverviewNodes.Custom;
  private searchString = '';

  @Mutation('enterCodeVault') enterCodeVault!: () => void;
  @Mutation('closeFiles') closeFiles!: () => void;
  @Getter('files') files!: ParsedFile[];

  private search(search: string) {
    this.searchString = search;
  }

  private shouldRender(button: string) {
    return button.toLowerCase().includes(this.searchString.toLowerCase());
  }

  private get renderedFiles() {
    return this.files.map((file) => ({
      filename: file.filename,
      functions: file.functions.filter((func) => this.shouldRender(func.name)),
    }));
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
