<template>
  <div id="contextual-menu">
    <div v-for="(editor, index) in editors" :key="index">
      <div>
        <div id="row">
          <VerticalMenuButton
            :label="editor.name"
            :onClick="() => switchEditor(editorType, index)"
            :isSelected="editorType === currEditorType && index === currEditorIndex">
          </VerticalMenuButton>
          <div class="buttons">
            <RenameEditorButton :editorType="editorType" :index="index" :oldName="editor.name"/>
            <DeleteEditorButton :editorType="editorType" :index="index" :name="editor.name"/>
          </div>
        </div>
      </div>
    </div>
    <VerticalMenuButton
      :label="'+'"
      :onClick="this.createNewEditor"
      :isSelected="false"
    />
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import VerticalMenuButton from '@/components/buttons/VerticalMenuButton.vue';
import { mapGetters } from 'vuex';
import EditorType from '@/EditorType';
import { Getter, Mutation } from 'vuex-class';
import { EditorModel } from '@/store/editors/types';
import { uniqueTextInput } from '@/inputs/prompt';
import RenameEditorButton from '@/components/buttons/RenameEditorButton.vue';
import DeleteEditorButton from '@/components/buttons/DeleteEditorButton.vue';
import { EditorSave, saveEditor } from '@/file/EditorAsJson';

@Component({
  components: { VerticalMenuButton, RenameEditorButton, DeleteEditorButton },
  computed: mapGetters([
    'currEditorType',
    'currEditorIndex',
  ]),
})
export default class NavbarContextualMenu extends Vue {
  @Prop({ required: true }) readonly editors!: EditorModel[];
  @Prop({ required: true }) readonly editorType!: EditorType;
  @Getter('editorNames') editorNames!: Set<string>;
  @Getter('currEditorModel') currEditorModel!: EditorModel;
  @Getter('overviewEditor') overviewEditor!: EditorModel;
  @Mutation('newEditor') newEditor!: (arg0: { editorType: EditorType; name: string }) => void;
  @Mutation('switchEditor') switch!: (arg0: { editorType: EditorType; index: number }) => void;
  @Mutation('updateNodeInOverview') readonly updateNodeInOverview!: (cEditor: EditorModel) => void;

  private createNewEditor(): void {
    const name: string | null = uniqueTextInput(this.editorNames,
      'Please enter a unique name for the editor');
    if (name !== null) {
      this.newEditor({ editorType: this.editorType, name });
      // New editor will be saved in periodic auto-save
    }
  }

  private switchEditor(editorType: EditorType, index: number) {
    // Save currEditorModel before switching as periodic save may not have captured last changes
    // and update overview editor if required
    this.updateNodeInOverview(this.currEditorModel);

    const oldEditorSaved: EditorSave = saveEditor(this.currEditorModel);
    const overviewEditorSave: EditorSave = saveEditor(this.overviewEditor);
    this.$cookies.set(`unsaved-editor-${this.currEditorModel.name}`, oldEditorSaved);
    this.$cookies.set('unsaved-editor-Overview', overviewEditorSave);

    this.switch({ editorType, index });
  }
}

</script>

<style lang="scss" scoped>
  #row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    &:hover {
      background: #1c1c1c;
      transition-duration: 0.1s;
      border-left-color: var(--blue);
      cursor: pointer;
    }
  }

  .buttons {
    display: flex;
  }

  #contextual-menu {
    // In order to have a higher z-index than IDE in CodeVault.
    z-index: 5;
    border: 1px solid var(--grey);
  }
</style>
