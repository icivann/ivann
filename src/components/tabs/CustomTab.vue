<template>
  <div class="search-tab h-100">
    <div
      class="msg"
      v-if="files.length === 0"
      @click="clickCodeVault"
    >
      Click Here to Add Custom Functions
    </div>
    <ExpandablePanel
      v-for="(file) in files"
      :key="file.filename"
      :name="file.filename"
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
import { Component, Vue } from 'vue-property-decorator';
import { mapGetters } from 'vuex';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import { CommonNodes } from '@/nodes/common/Types';
import { Mutation } from 'vuex-class';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
  computed: mapGetters(['files']),
})
export default class CustomTab extends Vue {
  private customNode: string = CommonNodes.Custom;

  @Mutation('enterCodeVault') enterCodeVault!: () => void;
  @Mutation('closeFiles') closeFiles!: () => void;

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
    padding-top: 10px;
    padding-bottom: 10px;
  }

  .msg:hover {
    background: #1c1c1c;
    transition-duration: 0.1s;
    cursor: pointer;
  }
</style>
