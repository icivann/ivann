<template>
  <div>
    <ExpandablePanel name="Models">
      <ButtonGrid>
        <AddNodeButton v-for="{ name } in editorList"
                       node="Dense"
                       :key="name"
                       :name="name"/>
      </ButtonGrid>
    </ExpandablePanel>
  </div>
</template>

<script lang="ts">
import { Component, Vue, Watch } from 'vue-property-decorator';
import ExpandablePanel from '@/components/ExpandablePanel.vue';
import AddNodeButton from '@/components/buttons/AddNodeButton.vue';
import ButtonGrid from '@/components/buttons/ButtonGrid.vue';
import { EditorModel } from '@/store/editors/types';

@Component({
  components: {
    ExpandablePanel,
    AddNodeButton,
    ButtonGrid,
  },
})
export default class ComponentsTab extends Vue {
  private editorList: EditorModel[] = this.$store.getters.modelEditors;

  @Watch('$store.getters.modelEditors')
  onEditorChange(editors: EditorModel[]) {
    this.editorList = editors;
  }
}
</script>
